"""
agent/tools/sqlite_tool.py
SQLite query execution and schema extraction - FULLY CORRECTED
"""

import sqlite3
import json
from typing import List, Dict, Tuple
from pathlib import Path


class SQLiteTool:
    """Tool for executing SQL queries against Northwind database"""
    
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.schema = self._extract_schema()
        print(f"✅ Connected to Northwind database")
    
    def _extract_schema(self) -> str:
        """Extract database schema as formatted text per assignment requirements"""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_parts = [
            "=== NORTHWIND DATABASE SCHEMA ===",
            "",
            "ASSIGNMENT REQUIREMENTS - SQL Guidelines:",
            "• Prefer Orders + \"Order Details\" + Products joins",
            "• Revenue: SUM(UnitPrice * Quantity * (1 - Discount)) from \"Order Details\"",
            "• Map categories via Categories join through Products.CategoryID",
            "",
            "CRITICAL TABLE NAMES:",
            "- Orders (OrderID, CustomerID, EmployeeID, OrderDate, ...)",
            "- \"Order Details\" (OrderID, ProductID, UnitPrice, Quantity, Discount) ⚠️ USE QUOTES!",
            "- Products (ProductID, ProductName, SupplierID, CategoryID, UnitPrice, ...)",
            "- Customers (CustomerID, CompanyName, Country, ...)",
            "- Categories (CategoryID, CategoryName, Description, ...)",
            "- Suppliers, Employees",
            "",
            "STANDARD JOIN PATTERNS:",
            "1. Product queries: Orders -> \"Order Details\" -> Products",
            "2. Category queries: Orders -> \"Order Details\" -> Products -> Categories",
            "3. Customer queries: Orders -> \"Order Details\" + Customers",
            "",
            "="*60,
            ""
        ]
        
        for table in tables:
            # Get table columns
            if ' ' in table:
                cursor.execute(f'PRAGMA table_info("{table}")')
            else:
                cursor.execute(f'PRAGMA table_info({table})')
            
            columns = cursor.fetchall()
            col_info = [f"{col[1]} ({col[2]})" for col in columns[:8]]  # First 8 cols
            
            # Get row count
            if ' ' in table:
                cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            else:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            
            if table == "Order Details":
                schema_parts.append(f"\n⚠️ \"{table}\" ({count} rows) - MUST USE QUOTES!")
            else:
                schema_parts.append(f"\n{table} ({count} rows)")
            
            schema_parts.append(f"  Columns: {', '.join(col_info)}")
            schema_parts.append("")
        
        return "\n".join(schema_parts)
    
    def get_schema(self) -> str:
        """Return formatted schema"""
        return self.schema
    
    def get_compact_schema(self) -> str:
        """Compact schema following assignment requirements"""
        cursor = self.conn.cursor()
        
        compact = [
            "NORTHWIND TABLES (per assignment):",
            "",
            "⚠️ Orders (OrderID, CustomerID, EmployeeID, OrderDate, ...)",
            "⚠️ \"Order Details\" (OrderID, ProductID, UnitPrice, Quantity, Discount) - USE QUOTES!",
            "⚠️ Products (ProductID, ProductName, SupplierID, CategoryID, UnitPrice, ...)",
            "⚠️ Customers (CustomerID, CompanyName, Country, ...)",
            "⚠️ Categories (CategoryID, CategoryName, Description, ...)",
            "   Suppliers, Employees",
            "",
            "ASSIGNMENT RULES:",
            "• Prefer: Orders + \"Order Details\" + Products joins",
            "• Revenue formula: SUM(od.UnitPrice * od.Quantity * (1 - od.Discount))",
            "• Category join: through Products.CategoryID",
            "",
            "STANDARD PATTERNS:",
            "FROM \"Order Details\" od",
            "JOIN Orders o ON od.OrderID = o.OrderID",
            "JOIN Products p ON od.ProductID = p.ProductID",
            "JOIN Categories c ON p.CategoryID = c.CategoryID  -- if needed",
        ]
        
        return "\n".join(compact)
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        return [row[0] for row in cursor.fetchall()]
    
    def execute_query(self, sql: str) -> Tuple[List[Dict], str]:
        """
        Execute SQL query (READ-ONLY)
        
        Returns:
            (results, error_message)
            results: List of dicts (rows)
            error_message: Empty string if successful, error message otherwise
        """
        try:
            # CRITICAL: Block all write operations (read-only database)
            sql_upper = sql.strip().upper()
            dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "REPLACE"]
            
            for keyword in dangerous_keywords:
                if sql_upper.startswith(keyword):
                    return [], f"Forbidden SQL operation: {keyword} (read-only database)"
            
            cursor = self.conn.cursor()
            cursor.execute(sql)
            
            # Check if query returns data
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                # Convert to list of dicts
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                return results, ""
            else:
                # Query doesn't return data (shouldn't happen with SELECT)
                return [], ""
        
        except sqlite3.Error as e:
            error_msg = str(e)
            
            # Add helpful hints for common errors
            if "no such table: OrderDetails" in error_msg:
                error_msg += ' | HINT: Use "Order Details" (with quotes and space) instead of OrderDetails'
            elif "no such table: order_details" in error_msg:
                error_msg += ' | HINT: Use "Order Details" (with quotes and space) instead of order_details'
            
            return [], error_msg
        except Exception as e:
            return [], f"Unexpected error: {str(e)}"
    
    def extract_tables_used(self, sql: str) -> List[str]:
        """
        Extract table names from SQL query (simple heuristic)
        Used for citation generation
        
        CRITICAL: Returns table names in correct format for citations
        Example: ["Orders", "Order Details", "Products"]
        NOT: ["orders", "order_details", "products"]
        """
        import re
        
        # Get all table names from database
        all_tables = self.get_table_names()
        
        # Find which tables are mentioned in the SQL
        tables_used = []
        sql_upper = sql.upper()
        
        for table in all_tables:
            # Check for table name with word boundaries
            # Handle both quoted ('Order Details') and unquoted (Orders) table names
            patterns = [
                r'\b' + re.escape(table.upper()) + r'\b',
                r"'" + re.escape(table.upper()) + r"'",
                r'"' + re.escape(table.upper()) + r'"'
            ]
            
            for pattern in patterns:
                if re.search(pattern, sql_upper):
                    # Add the properly formatted table name (with spaces, proper case)
                    tables_used.append(table)
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tables = []
        for table in tables_used:
            if table not in seen:
                seen.add(table)
                unique_tables.append(table)
        
        return unique_tables
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL without executing
        Returns: (is_valid, error_message)
        """
        # Basic validation
        sql_stripped = sql.strip().upper()
        
        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE']
        for keyword in dangerous_keywords:
            if keyword in sql_stripped.split():
                return False, f"Query contains forbidden keyword: {keyword}"
        
        # Check for common mistakes
        if 'ORDERDETAILS' in sql_stripped.replace(' ', ''):
            return False, 'Use "Order Details" (with quotes and space) instead of OrderDetails'
        
        if 'ORDER_DETAILS' in sql_stripped:
            return False, 'Use "Order Details" (with quotes and space) instead of ORDER_DETAILS'
        
        # Try EXPLAIN QUERY PLAN
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            return True, ""
        except sqlite3.Error as e:
            return False, str(e)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Cleanup"""
        try:
            self.close()
        except:
            pass


def clean_sql(sql: str) -> str:
    """
    Clean SQL query from markdown formatting
    
    LLMs often output SQL wrapped in markdown code blocks like:
    ```sql
    SELECT ...
    ```
    
    This function extracts just the SQL.
    """
    sql = sql.strip()
    
    # Remove markdown code blocks
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0]
    elif "```" in sql:
        sql = sql.split("```")[1].split("```")[0]
    
    # Remove extra whitespace
    sql = " ".join(sql.split())
    
    return sql.strip()


# Test function
if __name__ == "__main__":
    tool = SQLiteTool("../../data/northwind.sqlite")
    
    print("\n📋 Schema Preview:")
    print(tool.get_schema()[:800] + "...\n")
    
    print("\n📋 Compact Schema:")
    print(tool.get_compact_schema())
    
    print("\n📊 Tables:")
    print(tool.get_table_names())
    
    print("\n🔍 Test Query (correct):")
    sql = 'SELECT ProductName, UnitPrice FROM Products ORDER BY UnitPrice DESC LIMIT 5'
    results, error = tool.execute_query(sql)
    
    if error:
        print(f"❌ Error: {error}")
    else:
        print(f"✅ Retrieved {len(results)} rows")
        for row in results[:3]:
            print(f"  {row}")
    
    print("\n🔍 Test Query with Order Details (correct):")
    sql = 'SELECT * FROM "Order Details" LIMIT 3'
    results, error = tool.execute_query(sql)
    
    if error:
        print(f"❌ Error: {error}")
    else:
        print(f"✅ Retrieved {len(results)} rows")
    
    print("\n🔍 Test Query with wrong table name (should fail with hint):")
    sql = 'SELECT * FROM OrderDetails LIMIT 3'
    results, error = tool.execute_query(sql)
    
    if error:
        print(f"❌ Error (expected): {error}")
    
    print(f"\n📋 Tables used in query: {tool.extract_tables_used('SELECT * FROM Orders o JOIN \"Order Details\" od ON o.OrderID = od.OrderID')}")