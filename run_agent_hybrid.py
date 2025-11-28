"""
run_agent_hybrid.py
Main CLI runner for the hybrid agent - autograder-ready version
Loads id, question, format_hint from JSONL and writes outputs with id.

FIXED VERSION - All bugs corrected
"""

import argparse
import json
import dspy
from pathlib import Path
from typing import List, Dict, Any

from agent.graph_hybrid import HybridAgent


def load_questions(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file and preserve id + format_hint"""
    items = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Expect fields: id, question, format_hint
            item = {
                "id": data.get("id"),
                "question": data.get("question"),
                "format_hint": data.get("format_hint", "")
            }
            items.append(item)
    return items


def save_outputs(results: List[Dict], output_path: str):
    """Save results to JSONL file"""
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"\n✅ Results saved to {output_path}")


def run_single_query(agent: HybridAgent, question: str, format_hint: str = ""):
    """Run agent on single question and print formatted output"""
    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print(f"Format hint: {format_hint}")
    print("=" * 70)

    # For single queries, use a default ID
    result = agent.run("single_query", question, format_hint=format_hint)

    print(f"\n📝 Answer:")
    print(result.get('final_answer'))

    if result.get('sql'):
        print(f"\n💾 SQL Query:")
        print(result['sql'])

    print(f"\n📊 Confidence: {result['confidence']:.2f}")

    print(f"\n💡 Explanation:")
    print(result['explanation'])

    if result.get('citations'):
        print(f"\n📚 Citations:")
        for citation in result['citations']:
            print(f"  • {citation}")

    print("\n" + "=" * 70)


def run_batch(agent: HybridAgent, input_file: str, output_file: str):
    """Run agent on batch of questions from JSONL"""
    print(f"\n📂 Loading questions from {input_file}...")
    items = load_questions(input_file)
    print(f"✅ Loaded {len(items)} questions")

    print(f"\n🤖 Processing batch...")
    results = []
    for i, item in enumerate(items):
        qid = item.get("id")
        question = item.get("question", "")
        format_hint = item.get("format_hint", "")
        print(f"\n[{i+1}/{len(items)}] ({qid}) {question[:80]}...")
        
        # FIX: Pass qid as first argument
        out = agent.run(qid, question, format_hint=format_hint, thread_id=f"batch_{i}")
        
        # Ensure grader contract fields:
        out_record = {
            "id": qid,
            "final_answer": out.get("final_answer"),
            "sql": out.get("sql", ""),
            "confidence": out.get("confidence", 0.0),
            "explanation": out.get("explanation", ""),
            "citations": out.get("citations", [])
        }
        results.append(out_record)

    print(f"\n💾 Saving results...")
    save_outputs(results, output_file)

    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Total questions: {len(results)}")
    avg_conf = sum(r['confidence'] for r in results) / len(results) if results else 0.0
    print(f"  Avg confidence: {avg_conf:.2f}")

    sql_queries = sum(1 for r in results if r.get('sql'))
    print(f"  Questions with SQL: {sql_queries}/{len(results)}")

    avg_citations = sum(len(r.get('citations', [])) for r in results) / len(results) if results else 0.0
    print(f"  Avg citations per question: {avg_citations:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Local AI Agent - Hybrid RAG + SQL System (autograder-ready)"
    )

    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single question to ask the agent'
    )

    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to JSONL file with questions (one JSON object per line, with id,question,format_hint)'
    )

    parser.add_argument(
        '--out', '-o',
        type=str,
        default='outputs_hybrid.jsonl',
        help='Output file for batch results (default: outputs_hybrid.jsonl)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='phi3.5:3.8b-mini-instruct-q4_K_M',
        help='Ollama model to use (default: phi3.5:3.8b-mini-instruct-q4_K_M)'
    )

    parser.add_argument(
        '--db',
        type=str,
        default='data/northwind.sqlite',
        help='Path to Northwind SQLite database'
    )

    parser.add_argument(
        '--docs',
        type=str,
        default='docs',
        help='Path to documents directory'
    )

    args = parser.parse_args()

    # Setup DSPy with Phi-3.5
    print("🔧 Configuring DSPy with Ollama...")
    llm = dspy.LM(
    model=f"ollama/{args.model}",
    api_base="http://localhost:11434",
    max_tokens=1000,
    temperature=0.1
    )
    dspy.settings.configure(lm=llm)
    print(f"✅ Using model: {args.model}")

    # Initialize agent
    print("\n🤖 Initializing Hybrid Agent...")
    agent = HybridAgent(
        db_path=args.db,
        docs_dir=args.docs
    )

    # Run based on mode
    if args.query:
        # Single query mode: ask user to optionally provide format_hint (try to parse inline)
        format_hint = input("Optional format_hint (e.g. 'int' or '{category:str, quantity:int}'), leave blank for none: ").strip()
        run_single_query(agent, args.query, format_hint=format_hint)

    elif args.batch:
        # Batch mode
        if not Path(args.batch).exists():
            print(f"❌ Error: Input file not found: {args.batch}")
            return

        run_batch(agent, args.batch, args.out)

    else:
        # Interactive mode
        print("\n" + "=" * 70)
        print("Interactive Mode - Type 'quit' to exit")
        print("=" * 70)

        while True:
            try:
                question = input("\n🔍 Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                format_hint = input("Optional format_hint (press Enter to skip): ").strip()
                run_single_query(agent, question, format_hint=format_hint)

            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()