

import json
from typing import TypedDict, Annotated, List, Dict, Any
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.dspy_signatures import (
    RouterModule, PlannerModule, Text2SQLModule,
    SQLRepairModule, SynthesizerModule, ValidatorModule
)
from agent.rag.retrieval import SimpleRetriever
from agent.tools.sqlite_tool import SQLiteTool, clean_sql


class AgentState(TypedDict):
    """State that flows through the graph"""
    # Input
    question: str
    format_hint: str
    
    # Routing
    route: str  # "sql", "rag", "hybrid"
    
    # Planning
    date_range: str
    kpi_formula: str
    categories: str
    product_names: str
    
    # SQL execution
    sql_query: str
    sql_results: List[Dict]
    sql_error: str
    sql_tables_used: List[str]
    
    # Document retrieval
    doc_chunks: List[Dict]
    doc_citations: List[str]
    
    # Synthesis
    final_answer: Any
    confidence: float
    explanation: str
    
    # Citations (combined)
    citations: List[str]
    
    # Repair loop
    repair_count: int
    max_repairs: int
    
    # Validation
    is_valid: str
    validation_reason: str


class HybridAgent:
    """Main agent with LangGraph orchestration"""
    
    def __init__(
        self,
        db_path: str = "data/northwind.sqlite",
        docs_dir: str = "docs",
        checkpoint_dir: str = "checkpoints",
        use_optimized_sql: bool = True,
        optimized_sql_path: str = "data/optimized_text2sql.json"
    ):
        # Initialize tools
        self.sql_tool = SQLiteTool(db_path)
        self.retriever = SimpleRetriever(docs_dir)
        
        # Initialize DSPy modules
        self.router = RouterModule()
        self.planner = PlannerModule()
        self.text2sql = Text2SQLModule()
        self.sql_repairer = SQLRepairModule()
        self.synthesizer = SynthesizerModule()
        self.validator = ValidatorModule()
        
        # Load optimized Text2SQL if available
        self.use_optimized_sql = use_optimized_sql
        if use_optimized_sql and Path(optimized_sql_path).exists():
            try:
                self.text2sql.load(optimized_sql_path)
                print("✅ Loaded optimized Text2SQL module")
            except Exception as e:
                print(f"⚠️  Could not load optimized module: {e}")
                print("   Using baseline Text2SQL")
        
        # Build graph
        self.graph = self._build_graph()
        
        # Add checkpointer
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpointer = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        
        print("✅ HybridAgent initialized with LangGraph")
    
    def _build_graph(self) -> StateGraph:
        """Build the 8-node LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("nl2sql", self.nl2sql_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        workflow.add_node("validator", self.validator_node)
        workflow.add_node("repairer", self.repairer_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Router -> Planner (always)
        workflow.add_edge("router", "planner")
        
        # Planner -> conditional routing
        workflow.add_conditional_edges(
            "planner",
            self.route_after_planner,
            {
                "nl2sql": "nl2sql",
                "retriever": "retriever"
            }
        )
        
        # NL2SQL -> Executor
        workflow.add_edge("nl2sql", "executor")
        
        # Executor -> conditional (success or error)
        workflow.add_conditional_edges(
            "executor",
            self.route_after_executor,
            {
                "continue": "retriever",
                "repair": "repairer",
                "synthesize_sql_only": "synthesizer"
            }
        )
        
        # Repairer -> Executor (retry)
        workflow.add_edge("repairer", "executor")
        
        # Retriever -> Synthesizer
        workflow.add_edge("retriever", "synthesizer")
        
        # Synthesizer -> Validator
        workflow.add_edge("synthesizer", "validator")
        
        # Validator -> END
        workflow.add_edge("validator", END)
        
        return workflow
    
    # ========================================================================
    # NODE IMPLEMENTATIONS
    # ========================================================================
    
    def router_node(self, state: AgentState) -> AgentState:
        """Route question to appropriate path"""
        print(f"\n🔀 ROUTER: Analyzing question...")
        result = self.router(question=state["question"])
        route = result.route.lower().strip()
        print(f"   └─ Route chosen: {route}")
        
        # Validate route
        if route not in ["sql", "rag", "hybrid"]:
            q = state["question"].lower()
            if any(kw in q for kw in ["document", "policy", "calendar", "kpi"]):
                if any(kw in q for kw in ["top", "count", "total", "revenue"]):
                    route = "hybrid"
                else:
                    route = "rag"
            else:
                route = "sql"
        
        state["route"] = route
        print(f"🔍 ROUTER DECISION: '{route}' | question='{state['question']}'")
        state["repair_count"] = 0
        state["max_repairs"] = 2
        
        return state
    
    def planner_node(self, state: AgentState) -> AgentState:
        """Extract entities and constraints from question"""
        print(f"\n📋 PLANNER: Extracting entities...")
        result = self.planner(question=state["question"])
        
        # Extract fields safely with hasattr checks
        state["date_range"] = result.date_range if hasattr(result, 'date_range') and result.date_range.lower() != "none" else ""
        state["kpi_formula"] = result.kpi_formula if hasattr(result, 'kpi_formula') and result.kpi_formula.lower() != "none" else ""
        state["categories"] = result.categories if hasattr(result, 'categories') and result.categories.lower() != "none" else ""
        state["product_names"] = result.product_names if hasattr(result, 'product_names') and result.product_names.lower() != "none" else ""
        
        print(f"   ├─ Date range: {state['date_range'] or 'None'}")
        print(f"   ├─ KPI formula: {state['kpi_formula'] or 'None'}")
        print(f"   ├─ Categories: {state['categories'] or 'None'}")
        print(f"   └─ Products: {state['product_names'] or 'None'}")
        
        return state
    
    def nl2sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL query from natural language"""
        print(f"\n💾 NL2SQL: Generating SQL query...")
        
        # Build entity string
        entities = []
        if state.get("categories"):
            entities.append(f"Categories: {state['categories']}")
        if state.get("product_names"):
            entities.append(f"Products: {state['product_names']}")
        entities_str = ", ".join(entities) if entities else "None"
        
        # Get schema with reminders
        schema_with_reminder = f"""{self.sql_tool.get_compact_schema()}

CRITICAL REMINDERS:
- Table "Order Details" has a SPACE - MUST use: "Order Details" with quotes!
- For dates use: OrderDate LIKE '1997-06%' NOT YEAR(OrderDate) = 1997
- For revenue: SUM(od.UnitPrice * od.Quantity * (1-od.Discount)) from "Order Details"
- For categories: MUST JOIN Categories c ON p.CategoryID = c.CategoryID
"""
        
        result = self.text2sql(
            question=state["question"],
            db_schema=schema_with_reminder,
            date_range=state.get("date_range", ""),
            entities=entities_str
        )
        
        sql = getattr(result, 'sql', '')
        sql = clean_sql(sql)
        print(f"   └─ Generated: {sql[:100]}...")
        
        state["sql_query"] = sql
        return state
    
    def executor_node(self, state: AgentState) -> AgentState:
        """Execute SQL query"""
        print(f"\n⚡ EXECUTOR: Running SQL...")
        sql = state["sql_query"]
        print(f"   SQL: {sql[:150]}...")
        
        results, error = self.sql_tool.execute_query(sql)
        
        if error:
            print(f"   ❌ ERROR: {error}")
        else:
            print(f"   ✅ SUCCESS: Retrieved {len(results)} rows")
            if results:
                print(f"   └─ First row: {results[0]}")
        
        state["sql_results"] = results
        state["sql_error"] = error
        
        if not error:
            state["sql_tables_used"] = self.sql_tool.extract_tables_used(sql)
        
        return state
    
    def repairer_node(self, state: AgentState) -> AgentState:
        """Repair broken SQL with manual pattern-based fixes - COMPLETE VERSION"""
        
        if state["repair_count"] >= state["max_repairs"]:
            print(f"❌ Max repair attempts reached")
            state["sql_error"] = "Max repair attempts reached"
            return state
        
        sql = state["sql_query"]
        error = state["sql_error"]
        q_lower = state["question"].lower()
        
        print(f"🔧 REPAIRER: Attempt {state['repair_count'] + 1}/{state['max_repairs']}...")
        print(f"   Error was: {error}")
        
        # Manual fixes
        fixed_sql = sql
        import re
        

        if ("Query returned 0 rows" in error or not error) and len(state.get("sql_results", [])) == 0:
            print(f"   → Checking for date range issues in SQL...")
            
            # Get the date range from planner
            date_range = state.get("date_range", "")
            
            if date_range and " to " in date_range:
                print(f"   → Planner provided date range: {date_range}")
                
                # Parse dates
                parts = date_range.split(" to ")
                if len(parts) == 2:
                    start_date = parts[0].strip()
                    end_date = parts[1].strip()
                    
                    # Calculate next day for exclusive upper bound
                    from datetime import datetime, timedelta
                    try:
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        next_day = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                        
                        # Check if SQL has a WHERE clause with dates
                        if 'WHERE' in fixed_sql.upper():
                            # Replace existing date filter
                            # Pattern 1: BETWEEN dates
                            if 'BETWEEN' in fixed_sql.upper():
                                fixed_sql = re.sub(
                                    r"WHERE\s+.*?OrderDate\s+BETWEEN\s+'[^']+'\s+AND\s+'[^']+'",
                                    f"WHERE o.OrderDate >= '{start_date}' AND o.OrderDate < '{next_day}'",
                                    fixed_sql,
                                    flags=re.IGNORECASE
                                )
                                print(f"   → Replaced BETWEEN with: >= '{start_date}' AND < '{next_day}'")
                            
                            # Pattern 2: LIKE pattern
                            elif 'LIKE' in fixed_sql.upper():
                                fixed_sql = re.sub(
                                    r"WHERE\s+.*?OrderDate\s+LIKE\s+'[^']+'",
                                    f"WHERE o.OrderDate >= '{start_date}' AND o.OrderDate < '{next_day}'",
                                    fixed_sql,
                                    flags=re.IGNORECASE
                                )
                                print(f"   → Replaced LIKE with: >= '{start_date}' AND < '{next_day}'")
                            
                            else:
                                fixed_sql = re.sub(
                                    r"WHERE\s+.*?OrderDate\s*>=\s*'[^']+'\s+AND\s+.*?OrderDate\s*<=\s*'[^']+'",
                                    f"WHERE o.OrderDate >= '{start_date}' AND o.OrderDate < '{next_day}'",
                                    fixed_sql,
                                    flags=re.IGNORECASE
                                )
                                print(f"   → Updated date filter to: >= '{start_date}' AND < '{next_day}'")
                        else:
                            # No WHERE clause - add one before GROUP BY
                            if 'GROUP BY' in fixed_sql.upper():
                                insert_pos = fixed_sql.upper().find('GROUP BY')
                                date_clause = f" WHERE o.OrderDate >= '{start_date}' AND o.OrderDate < '{next_day}' "
                                fixed_sql = fixed_sql[:insert_pos] + date_clause + fixed_sql[insert_pos:]
                                print(f"   → Added WHERE clause: {date_clause}")
                    
                    except Exception as e:
                        print(f"   ⚠️  Date fix failed: {e}")
        

        if "no such table: OrderDetails" in error or "no such table: order_details" in error:
            print(f"   → Fixing: OrderDetails → \"Order Details\"")
            fixed_sql = re.sub(r'\bOrderDetails\b', '"Order Details"', fixed_sql, flags=re.IGNORECASE)
            fixed_sql = re.sub(r'\border_details\b', '"Order Details"', fixed_sql, flags=re.IGNORECASE)
        

        if "no such column: c.CategoryName" in error:
            print(f"   → Missing JOIN to Categories table")
            if "categories" not in fixed_sql.lower():
                products_join_pattern = r'(JOIN\s+Products\s+p\s+ON\s+[^\s]+\.[^\s]+\s*=\s*[^\s]+\.[^\s]+)'
                match = re.search(products_join_pattern, fixed_sql, re.IGNORECASE)
                
                if match:
                    insert_pos = match.end()
                    categories_join = ' JOIN Categories c ON p.CategoryID = c.CategoryID'
                    fixed_sql = fixed_sql[:insert_pos] + categories_join + fixed_sql[insert_pos:]
                    print(f"   → Added: JOIN Categories c ON p.CategoryID = c.CategoryID")
                else:
                    # Fallback template
                    print(f"   → Using category query template")
                    
                    # Get date range for template
                    date_range = state.get("date_range", "")
                    if date_range and " to " in date_range:
                        start, end = date_range.split(" to ")
                        start, end = start.strip(), end.strip()
                        from datetime import datetime, timedelta
                        try:
                            end_dt = datetime.strptime(end, '%Y-%m-%d')
                            next_day = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                            date_filter = f"WHERE o.OrderDate >= '{start}' AND o.OrderDate < '{next_day}'"
                        except:
                            date_filter = f"WHERE o.OrderDate BETWEEN '{start}' AND '{end}'"
                    else:
                        date_filter = ""
                    
                    fixed_sql = f"""
                    SELECT c.CategoryName, SUM(od.Quantity) AS total_quantity
                    FROM "Order Details" od
                    JOIN Orders o ON od.OrderID = o.OrderID
                    JOIN Products p ON od.ProductID = p.ProductID
                    JOIN Categories c ON p.CategoryID = c.CategoryID
                    {date_filter}
                    GROUP BY c.CategoryName
                    ORDER BY total_quantity DESC
                    LIMIT 1
                    """
        

        if "no such column: o.Discount" in error or "no such column: o.UnitPrice" in error or "no such column: o.Quantity" in error:
            print(f"   → Fixing column aliases: o.* → od.*")
            fixed_sql = re.sub(r'\bo\.UnitPrice\b', 'od.UnitPrice', fixed_sql, flags=re.IGNORECASE)
            fixed_sql = re.sub(r'\bo\.Quantity\b', 'od.Quantity', fixed_sql, flags=re.IGNORECASE)
            fixed_sql = re.sub(r'\bo\.Discount\b', 'od.Discount', fixed_sql, flags=re.IGNORECASE)
        

        if 'o."UnitPrice"' in fixed_sql or 'o."Quantity"' in fixed_sql:
            print(f"   → Removing quotes from column names")
            fixed_sql = re.sub(r'o\."UnitPrice"', 'od.UnitPrice', fixed_sql)
            fixed_sql = re.sub(r'o\."Quantity"', 'od.Quantity', fixed_sql)
            fixed_sql = re.sub(r'o\."Discount"', 'od.Discount', fixed_sql)
        

        if "no such function" in error.lower() or "YEAR(" in fixed_sql or "MONTH(" in fixed_sql:
            print(f"   → Removing unsupported date functions")
            fixed_sql = re.sub(r"YEAR\(([^)]+)\)\s*=\s*'?(\d+)'?", r"\1 LIKE '\2-%'", fixed_sql, flags=re.IGNORECASE)
            fixed_sql = re.sub(r"MONTH\(([^)]+)\)\s*=\s*'?(\d+)'?", r"\1 LIKE '____-\2-%'", fixed_sql, flags=re.IGNORECASE)
        

        if "misuse of aggregate" in error.lower():
            print(f"   → Fixing nested aggregates")
            if "AVG(" in fixed_sql and "SUM(" in fixed_sql:
                # Use AOV template
                date_range = state.get("date_range", "")
                if date_range and " to " in date_range:
                    start, end = date_range.split(" to ")
                    start, end = start.strip(), end.strip()
                    from datetime import datetime, timedelta
                    try:
                        end_dt = datetime.strptime(end, '%Y-%m-%d')
                        next_day = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                        date_filter = f"WHERE o.OrderDate >= '{start}' AND o.OrderDate < '{next_day}'"
                    except:
                        date_filter = f"WHERE o.OrderDate BETWEEN '{start}' AND '{end}'"
                else:
                    date_filter = "WHERE o.OrderDate LIKE '2016-%'"
                
                fixed_sql = f"""
                SELECT ROUND(
                    CAST(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS REAL) / 
                    COUNT(DISTINCT o.OrderID), 
                    2
                ) AS AOV
                FROM "Order Details" od
                JOIN Orders o ON od.OrderID = o.OrderID
                {date_filter}
                """
                print(f"   → Using AOV template with CAST")
        

        if 'syntax error' in error.lower() and '/' in fixed_sql:
            print(f"   → Division syntax error detected")
    
    # Check if it's an AOV query
            if 'aov' in q_lower or ('average' in q_lower and 'order' in q_lower) or 'AverageOrderValue' in fixed_sql:
                print(f"   → Replacing with AOV template")
            
            # Pattern 1: ROUND(SUM(...) / COUNT(...), digits)
                date_range = state.get("date_range", "")
                if date_range and " to " in date_range:
                    start, end = date_range.split(" to ")
                    start, end = start.strip(), end.strip()
                    from datetime import datetime, timedelta
                    try:
                        end_dt = datetime.strptime(end, '%Y-%m-%d')
                        next_day = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
                        date_filter = f"WHERE o.OrderDate >= '{start}' AND o.OrderDate < '{next_day}'"
                    except:
                        date_filter = f"WHERE o.OrderDate BETWEEN '{start}' AND '{end}'"
                else:
                    date_filter = ""
                
                # Use complete working template
                fixed_sql = f"""
                SELECT ROUND(
                    CAST(SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS REAL) / 
                    CAST(COUNT(DISTINCT o.OrderID) AS REAL), 
                    2
                ) AS AOV
                FROM "Order Details" od
                JOIN Orders o ON od.OrderID = o.OrderID
                {date_filter}
                """
                print(f"   → Used complete AOV template with date filter")
        
        # Clean whitespace
        fixed_sql = ' '.join(fixed_sql.split())
        
        print(f"   └─ Fixed SQL: {fixed_sql[:120]}...")
        
        state["sql_query"] = fixed_sql.strip()
        state["repair_count"] += 1
        
        return state
    
    def retriever_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents"""
        print(f"\n📚 RETRIEVER: Fetching documents...")
        
        if state["route"] in ["rag", "hybrid"]:
            docs = self.retriever.retrieve(state["question"], top_k=3)
            state["doc_chunks"] = docs
            state["doc_citations"] = self.retriever.format_citations(docs)
            print(f"   ✅ Retrieved {len(docs)} documents")
            for doc in docs:
                print(f"      - {doc['id']}")
        else:
            state["doc_chunks"] = []
            state["doc_citations"] = []
            print(f"   ⏭️  Skipped (route={state['route']})")
        
        return state
    
    def synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer from SQL results and documents"""
        print(f"\n🔬 SYNTHESIZER: Creating answer...")
        
        # Format SQL results
        sql_results = state.get("sql_results", [])
        if sql_results:
            sql_results_str = "SQL RESULTS:\n"
            for i, row in enumerate(sql_results[:20]):
                sql_results_str += f"Row {i+1}: {row}\n"
            print(f"   ├─ Using {len(sql_results)} SQL rows")
        else:
            sql_results_str = "NO SQL RESULTS"
            print(f"   ├─ No SQL results")
        
        # Format documents
        doc_chunks_list = state.get("doc_chunks", [])
        if doc_chunks_list:
            doc_chunks_str = "DOCUMENTS:\n"
            for doc in doc_chunks_list:
                doc_chunks_str += f"[{doc['id']}]: {doc['content'][:200]}...\n"
            print(f"   ├─ Using {len(doc_chunks_list)} documents")
        else:
            doc_chunks_str = "NO DOCUMENTS"
            print(f"   ├─ No documents")
        
        format_hint = state.get("format_hint", "")
        print(f"   └─ Format required: {format_hint}")
        
        result = self.synthesizer(
            question=state["question"],
            sql_results=sql_results_str,
            doc_chunks=doc_chunks_str,
            format_hint=format_hint
        )
        
        # Parse final answer
        final_answer = result.final_answer
        
        # Try to parse JSON if format suggests it
        if "{" in format_hint or "[" in format_hint:
            import json
            if isinstance(final_answer, str):
                try:
                    final_answer = json.loads(final_answer)
                except:
                    # Try to extract JSON from text
                    import re
                    match = re.search(r'\{[^{}]*\}|\[[^\[\]]*\]', final_answer)
                    if match:
                        try:
                            final_answer = json.loads(match.group(0))
                        except:
                            pass
        
        state["final_answer"] = final_answer
        state["explanation"] = result.explanation
        
        # Parse confidence
        try:
            confidence = float(result.confidence)
            state["confidence"] = max(0.0, min(1.0, confidence))
        except:
            state["confidence"] = 0.5
        
        print(f"   Result: {str(state['final_answer'])[:100]}")
        print(f"   Confidence: {state['confidence']}")
        
        # Parse citations
        citations = []
        
        # Model-supplied citations
        raw_cites = getattr(result, "citations", None)
        if raw_cites:
            if isinstance(raw_cites, list):
                citations.extend(raw_cites)
            elif isinstance(raw_cites, str):
                # Parse comma-separated string
                for cite in raw_cites.split(','):
                    cite = cite.strip()
                    if cite:
                        citations.append(cite)
        
        # SQL table citations
        for table in state.get("sql_tables_used", []):
            if table not in citations:
                citations.append(table)
        
        # Document citations
        for doc in state.get("doc_citations", []):
            if doc not in citations:
                citations.append(doc)
        
        # Remove duplicates while preserving order
        seen = set()
        final_cites = []
        for c in citations:
            if c and c not in seen:
                seen.add(c)
                final_cites.append(c)
        
        state["citations"] = final_cites
        
        return state
    
    def validator_node(self, state: AgentState) -> AgentState:
        """Validate final answer"""
        print(f"\n✔️  VALIDATOR: Checking answer...")
        
        result = self.validator(
            question=state["question"],
            answer=str(state.get("final_answer", "")),
            sql_query=state.get("sql_query", ""),
            sql_error=state.get("sql_error", ""),
            format_hint=state.get("format_hint", "")
        )
        
        state["is_valid"] = result.is_valid.lower()
        state["validation_reason"] = result.reason
        
        print(f"   ├─ Valid: {state['is_valid']}")
        print(f"   └─ Reason: {result.reason[:80]}...")
        
        return state
    

    
    def route_after_planner(self, state: AgentState) -> str:
        """Route after planning based on route decision"""
        route = state["route"]
        
        # For RAG-only, skip SQL entirely
        if route == "rag":
            return "retriever"
        # For SQL and hybrid, go to SQL first
        else:
            return "nl2sql"
    
    def route_after_executor(self, state: AgentState) -> str:
        """Route after SQL execution - FIXED to catch empty results"""
        
        # Priority 1: If SQL error and repairs left → repair
        if state["sql_error"] and state["repair_count"] < state["max_repairs"]:
            return "repair"
        
        # Priority 2: If SQL succeeded but returned 0 rows when data expected → repair
        results = state.get("sql_results", [])
        
        if not state["sql_error"] and len(results) == 0:
            q_lower = state["question"].lower()
            
            # Keywords that indicate we should have found data
            expects_data_keywords = [
                'top', 'highest', 'best', 'total', 'revenue', 
                'during', 'summer', 'winter', 'category', 'customer', 'aov'
            ]
            
            expects_data = any(kw in q_lower for kw in expects_data_keywords)
            
            if expects_data and state["repair_count"] < state["max_repairs"]:
                print(f"   ⚠️  SQL returned 0 rows but data expected - triggering repair")
                state["sql_error"] = "Query returned 0 rows (possible date mismatch)"
                return "repair"
        
        # Priority 3: Normal routing
        route = state.get("route", "")
        if route == "hybrid":
            return "continue"
        elif route == "sql":
            return "synthesize_sql_only"
        else:
            return "continue"
    
    
    def run(self, q_id: str, question: str, format_hint: str = "", thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the agent on a single question
        
        Args:
            q_id: Question ID from batch file
            question: Natural language question
            format_hint: Expected output format
            thread_id: Thread ID for checkpointing
        
        Returns formatted output matching required schema
        """
        initial_state = AgentState(
            question=question,
            format_hint=format_hint,
            route="",
            date_range="",
            kpi_formula="",
            categories="",
            product_names="",
            sql_query="",
            sql_results=[],
            sql_error="",
            sql_tables_used=[],
            doc_chunks=[],
            doc_citations=[],
            final_answer="",
            confidence=0.0,
            explanation="",
            citations=[],
            repair_count=0,
            max_repairs=2,
            is_valid="",
            validation_reason=""
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.compiled_graph.invoke(initial_state, config)
        
        # Format output according to required schema
        return {
            "id": q_id,
            "final_answer": final_state.get("final_answer", ""),
            "sql": final_state.get("sql_query", ""),
            "confidence": final_state.get("confidence", 0.0),
            "explanation": final_state.get("explanation", ""),
            "citations": final_state.get("citations", [])
        }
    
    def run_batch(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Run agent on multiple questions from JSONL batch file
        
        Args:
            questions: List of dicts with keys: 'id', 'question', 'format_hint'
        """
        results = []
        for i, q_data in enumerate(questions):
            if isinstance(q_data, str):
                q_id = f'q_{i}'
                question = q_data
                format_hint = ''
            else:
                q_id = q_data.get('id', f'q_{i}')
                question = q_data['question']
                format_hint = q_data.get('format_hint', '')
            
            print(f"\n[{i+1}/{len(questions)}] ({q_id}) {question[:80]}...")
            result = self.run(q_id, question, format_hint, thread_id=f"batch_{i}")
            results.append(result)
        
        return results


# Test
if __name__ == "__main__":
    import dspy
    
    # Configure DSPy with Phi-3.5
    llm = dspy.LM(
        model="ollama/phi3.5:3.8b-mini-instruct-q4_K_M",
        api_base="http://localhost:11434",
        max_tokens=1000,
        temperature=0.1
    )
    dspy.settings.configure(lm=llm)
    
    # Create agent
    agent = HybridAgent()
    
    # Test query
    result = agent.run(
        q_id="test_1",
        question="What are the top 5 products by revenue?",
        format_hint="list[{product: str, revenue: float}]"
    )
    print(json.dumps(result, indent=2))
