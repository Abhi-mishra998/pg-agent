#!/usr/bin/env python3
"""
TerminalAgent - pg-agent CLI Interface

Responsibilities:
- Interactive command-line interface
- Pipeline orchestration via core.Pipeline
- Load files (logs / JSON / pgbench)
- Run Signal → Evidence → Recommendation pipeline
- Use KB for similar incidents and best practices
- Use LLM ONLY for explanation (not decisions)
- Generate reports

Safe for client usage.
"""

import cmd
import logging
import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional clipboard support for one-click copy to Slack
try:
    import pyperclip
    _HAS_PYPERCLIP = True
except Exception:
    _HAS_PYPERCLIP = False

from dotenv import load_dotenv
load_dotenv()

# ---------------- COLOR SUPPORT ----------------

try:
    from colorlog import ColoredFormatter
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False


# ANSI color codes
class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'


def colorize(text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    color_code = getattr(Colors, color.upper(), Colors.RESET)
    bold_code = Colors.BOLD if bold else ''
    return f"{bold_code}{color_code}{text}{Colors.RESET}"


def format_severity(severity: str) -> str:
    """Format severity with appropriate color."""
    severity = severity.upper()
    colors_map = {
        'CRITICAL': ('RED', True),
        'HIGH': ('YELLOW', True),
        'MEDIUM': ('CYAN', False),
        'LOW': ('GREEN', False),
    }
    color, bold = colors_map.get(severity, ('WHITE', False))
    return colorize(severity, color, bold)


# =====================================================================
# INTERNAL IMPORTS
# =====================================================================

from loader.file_loader import FileLoader
from parser.input_parser import InputParser
from llm.llama_client import LlamaClient
from agent.validator import OutputValidator
from agent.runbook_agent import RunbookAgent, RunbookOutput
from core.pipeline import Pipeline, create_default_pipeline
from reports.report_generator import ReportGenerator


# =====================================================================
# TERMINAL AGENT
# =====================================================================

class TerminalAgent(cmd.Cmd):
    """
    pg-agent Interactive Terminal Interface.
    
    Commands:
        analyze <text | file>   Analyze logs / SQL / report
        ask <q>                Ask PostgreSQL question
        search <query>         Search KB for similar issues
        similar <kb_id>        Find similar KB entries
        report <topic>         Generate HTML report
        status                 System status
        history                Conversation history
        clear                  Clear history
        quit                   Exit
    """

    prompt = "pg-agent> "
    intro = """
╔══════════════════════════════════════════════════════╗
║                    pg-agent                          ║
║     Postgres Incident Intelligence Agent             ║
╠══════════════════════════════════════════════════════╣
║  analyze <text | file>   Analyze logs / SQL / report ║
║  ask <q>                Ask PostgreSQL question     ║
║  search <query>         Search KB for similar issues║
║  similar <kb_id>        Find similar KB entries     ║
║  report <topic>         Generate HTML report        ║
║  runbook <q>            Generate structured runbook ║
║  status                 System status               ║
║  history                Conversation history        ║
║  clear                  Clear history               ║
║  quit                   Exit                        ║
╚══════════════════════════════════════════════════════╝
"""

    # ------------------------------------------------
    # Init
    # ------------------------------------------------

    def __init__(
        self,
        input_parser: Optional[InputParser] = None,
        output_dir: str = "data/output",
        log_level: str = "INFO",
        kb_path: str = "data/kb_unified.json",
    ):
        super().__init__()

        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        self.logger = logging.getLogger(self.__class__.__name__)

        # Core components
        self.input_parser = input_parser or InputParser()
        self.file_loader = FileLoader()
        
        # Pipeline (core orchestration)
        self.pipeline = create_default_pipeline(log_level=log_level)
        
        # LLM client (silent mode to avoid error logs when unavailable)
        self.llm = LlamaClient(silent=True)
        self.validator = OutputValidator()
        
        # Reporter
        self.reporter = ReportGenerator()

        # Runbook agent (lazy initialization)
        self.runbook_agent: Optional[RunbookAgent] = None

        # Runtime state
        self.output_dir = output_dir
        self.session_start = datetime.utcnow()
        self.history: List[Dict[str, str]] = []

        self.logger.info("TerminalAgent initialized")

    # ------------------------------------------------
    # Lifecycle
    # ------------------------------------------------

    def run(self) -> None:
        """Start the terminal agent."""
        self.logger.info("Starting pg-agent terminal")

        if not self.llm.is_available():
            print("⚠️ Ollama not running (LLM explanations disabled)")
            print("   Run: ollama serve\n")

        try:
            self.cmdloop()
        except KeyboardInterrupt:
            print("\n👋 Exiting pg-agent")

    # ------------------------------------------------
    # Processing Methods
    # ------------------------------------------------

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a single query through the pipeline.
        
        Args:
            query: Natural language query or SQL
            
        Returns:
            Dictionary with results
        """
        self.logger.info(f"Processing query: {query[:50]}...")
        
        result = self.pipeline.run_simple(query)
        
        # Add to history
        self._remember("query", query, result)
        
        return result

    def process_input(self, input_data: Any) -> List[Dict[str, Any]]:
        """
        Process input data through the pipeline.
        
        Args:
            input_data: Raw data (string, dict, file path)
            
        Returns:
            List of results
        """
        results = []
        
        if isinstance(input_data, str):
            # Check if it's a file path
            path = Path(input_data)
            if path.exists() and path.is_file():
                file_ctx = self.file_loader.load(input_data)
                input_data = file_ctx["content"]
                self.logger.info(f"Loaded file: {file_ctx['path']}")
        
        if isinstance(input_data, str):
            # Process as text
            result = self.pipeline.run_simple(input_data)
            results.append(result)
            self._remember("input", input_data, result)
        
        elif isinstance(input_data, list):
            # Process each item
            for item in input_data:
                if isinstance(item, str):
                    result = self.pipeline.run_simple(item)
                    results.append(result)
                elif isinstance(item, dict):
                    result = self.pipeline.run_simple(item)
                    results.append(result)
        
        elif isinstance(input_data, dict):
            # Process as dict
            result = self.pipeline.run_simple(input_data)
            results.append(result)
            self._remember("input", str(input_data)[:100], result)
        
        else:
            self.logger.warning(f"Unknown input type: {type(input_data)}")
        
        return results

    # ------------------------------------------------
    # Commands
    # ------------------------------------------------

    def do_analyze(self, args: str) -> None:
        """
        analyze <text | file_path>
        
        Analyzes input and provides recommendations with KB context.
        Also generates structured runbook output when LLM is available.
        """
        if not args.strip():
            print("Usage: analyze <sql | log | file_path>")
            return

        raw_input = args.strip()
        
        # 1️⃣ Load file OR raw text
        path = Path(raw_input)
        if path.exists() and path.is_file():
            try:
                file_ctx = self.file_loader.load(raw_input)
                raw_data = file_ctx["content"]
                print(f"📂 Loaded file: {file_ctx['path']} ({file_ctx['type']})")
            except Exception as e:
                print(f"❌ File load failed: {e}")
                return
        else:
            parsed = self.input_parser.parse_query(raw_input)
            raw_data = parsed["data"]

        # 2️⃣ Run pipeline
        result = self.pipeline.run_simple(raw_data)

        # 3️⃣ Output results
        self._display_analysis_results(result)
        
        # 4️⃣ Generate runbook output if LLM is available
        self._generate_and_display_runbook(raw_input, result)
        
        self._remember("analyze", raw_input, result)

    def _generate_and_display_runbook(self, raw_input: str, pipeline_result: Dict) -> None:
        """Generate and display runbook output from pipeline result."""
        # Initialize runbook agent if needed
        if self.runbook_agent is None:
            try:
                self.runbook_agent = RunbookAgent()
            except Exception:
                # Runbook agent not available, skip
                return
        
        # Check if LLM is available
        if not self.runbook_agent.is_available():
            return
        
        # Run full pipeline to get structured result for runbook
        try:
            full_result = self.pipeline.run(raw_input)
            
            if full_result.success:
                runbook_output = self.runbook_agent.generate_from_pipeline_result(
                    pipeline_result=full_result,
                    user_question=f"Analyze this PostgreSQL issue: {raw_input[:200]}...",
                )
                self._display_runbook_output(runbook_output)
        except Exception as e:
            self.logger.warning(f"Runbook generation failed: {e}")

    def do_search(self, args: str) -> None:
        """
        search <query>
        
        Search the knowledge base for similar issues.
        """
        if not args.strip():
            print("Usage: search <query>")
            return

        query = args.strip()
        print(f"\n🔍 Searching KB for: '{query}'")
        print("-" * 50)
        
        # Use recommender's search
        from recommendations.pg_recommender import PgRecommender
        recommender = PgRecommender()
        
        report = recommender.recommend_from_query(query)
        
        if not report.recommendations:
            print("No matching KB entries found.")
            return

        print(f"Found {len(report.recommendations)} matching entries:\n")

        for i, r in enumerate(report.recommendations[:5], 1):
            print(f"{i}. [{r.severity.upper()}] {r.title}")
            print(f"   {r.description[:80]}...")
            if r.kb_entry_id:
                print(f"   KB: {r.kb_entry_id}")
            print()

    def do_similar(self, args: str) -> None:
        """
        similar <kb_id>
        
        Find KB entries similar to a given KB entry.
        """
        if not args.strip():
            print("Usage: similar <kb_id>")
            return

        kb_id = args.strip()
        print(f"\n🔗 Finding entries similar to: {kb_id}")
        print("-" * 50)
        
        from recommendations.pg_recommender import PgRecommender
        recommender = PgRecommender()
        
        similar = recommender.get_similar_entries(kb_id, top_k=5)

        if not similar:
            print(f"No similar entries found for '{kb_id}'")
            return

        print(f"\nSimilar entries:\n")

        for i, result in enumerate(similar, 1):
            print(f"{i}. {result.entry.metadata.kb_id}")
            print(f"   {result.entry.problem_identity.issue_type[:60]}...")
            print(f"   Score: {result.score:.3f}")
            print()

    def do_ask(self, args: str) -> None:
        """
        ask <question>
        
        Ask a PostgreSQL question with KB context.
        """
        if not args.strip():
            print("Usage: ask <question>")
            return

        query = args.strip()
        
        # KB-based answers
        from recommendations.pg_recommender import PgRecommender
        recommender = PgRecommender()
        
        report = recommender.recommend_from_query(query)
        
        if report.recommendations:
            print("\n📚 KB-BASED ANSWERS:")
            print("-" * 50)
            for r in report.recommendations[:5]:
                print(f"\n🔹 {r.title}")
                print(f"   {r.description}")
                if r.kb_entry_id:
                    print(f"   Source: {r.kb_entry_id}")

        # LLM response
        if self.llm.is_available():
            response = self.llm.generate(
                prompt=args,
                system="You are a PostgreSQL expert."
            )
            print("\n💬", response.text)
        else:
            print("\n💬 LLM not available. KB-based answers shown above.")

        self._remember("ask", query)

    def do_report(self, args: str) -> None:
        """
        report <topic>
        
        Generate an HTML report for the given topic.
        Topics: incident, review, kb, full
        """
        topic = (args or "").strip().lower()
        
        if not topic:
            print("Usage: report <topic>")
            print("Topics: incident, review, kb, full")
            return
        
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        if topic == "incident":
            if not self.history:
                print("⚠️ No analysis history. Run 'analyze' first.")
                return
            
            # Get last analysis
            for entry in reversed(self.history):
                if entry.get("role") == "analyze":
                    raw = entry.get("content", "")
                    pipeline_result = self.pipeline.run(raw)
                    
                    if pipeline_result.signal_result and pipeline_result.evidence_collection:
                        report_html = self.reporter.generate_incident_report(
                            signal_result=pipeline_result.signal_result,
                            evidence=pipeline_result.evidence_collection,
                            output_dir=str(output_dir)
                        )
                        print(f"\n📊 INCIDENT REPORT GENERATED")
                        print(f"   Path: {report_html}")
                        break
            else:
                print("⚠️ No analyze entries found.")
        
        elif topic == "review":
            from recommendations.review_schema import ReviewCard, ReviewAction
            from recommendations.review_renderer import render_to_html
            from recommendations.review_schema import (
                ActionRisk, OperationMode, ImpactScope,
                create_risk_indicators, create_safety_warnings,
                RollbackPlan,
            )
            
            card = ReviewCard(
                card_id=f"rc_{timestamp}",
                title="Sample Review Card",
                summary="Generated from report command",
                category="sample",
                severity="medium",
            )
            
            action = ReviewAction(
                action_id="act_sample",
                action_type="CREATE_INDEX",
                title="Create sample index",
                description="Sample action",
                sql_command="CREATE INDEX CONCURRENTLY idx_sample ON table_name(column);",
                risk=ActionRisk.LOW.value,
                risk_indicators=create_risk_indicators(ActionRisk.LOW.value),
                safety_warnings=create_safety_warnings("CREATE INDEX...", ActionRisk.LOW.value),
                operation_mode=OperationMode.CONCURRENTLY.value,
                rollback_plan=RollbackPlan(
                    rollback_command="DROP INDEX CONCURRENTLY idx_sample;",
                ),
            )
            card.actions = [action]
            
            report_html = render_to_html(card)
            report_path = output_dir / f"review_report_{timestamp}.html"
            report_path.write_text(report_html)
            
            print(f"\n📋 REVIEW REPORT GENERATED")
            print(f"   Path: {report_path}")
        
        elif topic == "kb":
            from recommendations.kb_loader import KBLoader
            
            kb_loader = KBLoader()
            kb = kb_loader.load_file("data/kb_unified.json")
            
            if not kb:
                print("⚠️ Knowledge base not loaded.")
                return
            
            stats = kb_loader.get_statistics(kb)
            report_path = output_dir / f"kb_report_{timestamp}.html"
            report_path.write_text(f"""
<html><head><title>KB Report</title></head>
<body>
<h1>📚 Knowledge Base Report</h1>
<p>Generated: {datetime.utcnow().isoformat()}</p>
<p>Total Entries: {kb.entry_count}</p>
<p>Categories: {len(stats['categories'])}</p>
</body></html>
""")
            
            print(f"\n📚 KB REPORT GENERATED")
            print(f"   Path: {report_path}")
            print(f"   Total Entries: {kb.entry_count}")
        
        else:
            print(f"⚠️ Unknown topic: {topic}")
            print("Available: incident, review, kb, full")

    def do_status(self, args: str) -> None:
        """Show system status."""
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        print(f"LLM Available     : {self.llm.is_available()}")
        print(f"Session Start     : {self.session_start}")
        print(f"History Size      : {len(self.history)}")
        print(f"Output Directory  : {self.output_dir}")

    def do_history(self, args: str) -> None:
        """Show conversation history."""
        for h in self.history[-10:]:
            print(f"[{h.get('role', 'unknown')}] {h.get('time', '')}")

    def do_clear(self, args: str) -> None:
        """Clear conversation history."""
        self.history.clear()
        print("🧹 History cleared")

    def do_runbook(self, args: str) -> None:
        """
        runbook <question | analysis_output>
        
        Generate structured runbook response for PostgreSQL issues.
        Uses RAG with knowledge base and deterministic analysis.
        
        Examples:
          runbook 'Query is slow due to sequential scan'
          runbook '{"seq_scan_ratio": 0.99, "table_rows": 10000000}'
          runbook 'How to diagnose blocking transactions'
        """
        if not args.strip():
            print("Usage: runbook <question | metrics_json>")
            print("\nExamples:")
            print("  runbook 'Query is slow due to sequential scan'")
            print("  runbook '{\"seq_scan_ratio\": 0.99, \"table_rows\": 10000000}'")
            print("  runbook 'How to diagnose blocking transactions'")
            return
        
        # Initialize runbook agent if needed
        if self.runbook_agent is None:
            try:
                self.runbook_agent = RunbookAgent()
            except Exception as e:
                print(f"Error initializing RunbookAgent: {e}")
                return
        
        # Check if LLM is available
        if not self.runbook_agent.is_available():
            print("⚠️ Ollama not running. Runbook AI requires Ollama.")
            print("   Run: ollama serve")
            return
        
        # Check if input is JSON (evidence)
        try:
            evidence = json.loads(args)
            runbook_output = self.runbook_agent.generate_from_query(
                query="Analyze PostgreSQL performance issue",
                evidence=evidence,
            )
        except json.JSONDecodeError:
            # Treat as natural language query
            runbook_output = self.runbook_agent.generate_from_query(
                query=args,
            )
        
        # Display output
        self._display_runbook_output(runbook_output)
        
        self._remember("runbook", args)

    def _display_runbook_output(self, output: RunbookOutput) -> None:
        """Display runbook output in terminal."""
        print("\n" + "=" * 60)
        print("📋 RUNBOOK OUTPUT")
        print("=" * 60)
        
        if output.diagnosis:
            print(f"\n🔍 DIAGNOSIS\n{output.diagnosis}")
        else:
            print("\n🔍 DIAGNOSIS\nInsufficient data to conclude")
        
        if output.evidence:
            print(f"\n📊 EVIDENCE\n{output.evidence}")
        
        if output.root_cause:
            print(f"\n🎯 ROOT CAUSE\n{output.root_cause}")
        
        if output.immediate_actions:
            print(f"\n⚡ IMMEDIATE ACTIONS\n{output.immediate_actions}")
        
        if output.preventive_actions:
            print(f"\n🛡️ PREVENTIVE ACTIONS\n{output.preventive_actions}")
        
        if output.cli_commands:
            print(f"\n💻 CLI COMMANDS\n{output.cli_commands}")
        
        print("\n" + "=" * 60)

    def do_quit(self, args: str) -> bool:
        """Exit the agent."""
        print("👋 Bye")
        return True

    do_exit = do_quit
    do_q = do_quit

    # ------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------

    def _remember(self, role: str, content: str, result: Dict = None) -> None:
        """Add to conversation history."""
        entry = {
            "role": role,
            "content": content,
            "time": datetime.utcnow().isoformat(),
        }
        if result:
            entry["result"] = result
        self.history.append(entry)

    def _display_analysis_results(self, result: Dict[str, Any]) -> None:
        """Display analysis results in terminal format."""
        print("\n🧠 INCIDENT ANALYSIS")
        print("-" * 50)

        if result.get("signals"):
            for s in result["signals"]:
                severity = s.get("severity", "unknown")
                print(f"🚨 [{format_severity(severity)}] {s.get('name', 'unknown')}")
        else:
            print("✅ No critical signals detected")

        print("\n📊 EVIDENCE SUMMARY")
        print(f"• Evidence count   : {result.get('evidence_count', 0)}")
        print(f"• Confidence       : {result.get('confidence', 0)}")

        print("\n🛠️ EXPERT RECOMMENDATIONS")
        recommendations = result.get("recommendations", [])
        if recommendations:
            for r in recommendations:
                print(f"\n🔹 {r.get('title', 'Unknown')}")
                print(f"   {r.get('description', '')[:100]}")
        else:
            print("✅ No action required")

        print(f"\n📈 Risk Level: {result.get('risk_level', 'unknown')}")
        print(f"⏱️  Processing Time: {result.get('processing_time_ms', 0):.2f}ms")

        if result.get("error"):
            print(f"\n❌ Error: {result['error']}")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    TerminalAgent().run()

