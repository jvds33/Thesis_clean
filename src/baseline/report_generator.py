import json
import os
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_json_summary(summary_data, output_dir, filename):
    """Saves the summary data as a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"JSON summary saved to {filepath}")

def generate_report(run_data):
    """
    Generates a PDF report and a JSON summary from the given run data.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain = run_data['config']['domain']
    model_name = run_data['config']['model']
    shot = f"{run_data['config']['few_shot']}-shot"
    
    # Check if custom output paths are provided
    if 'output_paths' in run_data:
        json_filepath = run_data['output_paths']['summary']
        pdf_filepath = run_data['output_paths']['pdf']
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(pdf_filepath), exist_ok=True)
    
    if 'output_paths' not in run_data:
        # Use default naming and directory
        pdf_filename = f"report_{domain}_{model_name}_{shot}_{timestamp}.pdf"
        json_filename = f"summary_{domain}_{model_name}_{shot}_{timestamp}.json"
        output_dir = "outputs"
        
        json_filepath = os.path.join(output_dir, json_filename)
        pdf_filepath = os.path.join(output_dir, pdf_filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    # Save JSON summary
    with open(json_filepath, 'w') as f:
        json.dump(run_data, f, indent=2)
    print(f"JSON summary saved to {json_filepath}")

    # Generate PDF
    print(f"Generating PDF report: {os.path.basename(pdf_filepath)}")
    pdf = PDFReport(run_data['config'])
    pdf.add_page()
    pdf.generate_content(run_data)
    pdf.output(pdf_filepath)
    print(f"PDF report saved to {pdf_filepath}")

def create_report(predictions_file, data, sources, token_usage):
    """
    This is a wrapper function to be compatible with the run_extractor_agents.py script.
    It reformats the data and calls the generate_report function.
    """
    
    # Reformat the data to match the expected input of generate_report
    run_data = {
        "config": {
            "model": data['args']['model'],
            "voters": "N/A",  # This information is not available in the new data format
            "few_shot": data['args']['prompt_type'],
            "domain": data['args']['dataset'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "agent_level": {
            "aoe": {"precision": 0, "recall": 0, "f1": 0},  # Placeholder
            "sentiment": {"accuracy": 0, "macro_f1": 0, "confusion_matrix": [], "labels": []},  # Placeholder
            "category": {"accuracy": 0, "macro_f1": 0, "top_errors_absolute": [], "top_errors_relative": []}  # Placeholder
        },
        "pipeline_level": {
            "exact_match_f1": data['scores']['f1'],
            "partial_match_f1": 0  # Placeholder
        },
        "cost": {
            "total_cost_usd": 0, # Placeholder
        },
        "errors": {
            "extraction": 0, # Placeholder
            "sentiment": 0, # Placeholder
            "category": 0 # Placeholder
        },
        "case_studies": [], # Placeholder
        "output_paths": {
            "summary": predictions_file.replace(".json", "_summary.json"),
            "pdf": predictions_file.replace(".json", ".pdf")
        }
    }
    
    if token_usage:
        # This is a simplified version of the cost calculation.
        # The original logic was in the deleted report_generator.
        total_prompt = sum(v.get('prompt_tokens', 0) for v in token_usage.values())
        total_completion = sum(v.get('completion_tokens', 0) for v in token_usage.values())
        
        run_data["cost"]["detailed_tokens"] = {
            "unified": token_usage.get("unified", {}),
            "sentiment": token_usage.get("sentiment", {}),
            "category": token_usage.get("category", {}),
            "totals": {
                "input_tokens": total_prompt,
                "output_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "cost": 0 # Placeholder
            }
        }
        
    generate_report(run_data)

class PDFReport(FPDF):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.set_auto_page_break(auto=True, margin=15)
        self.set_font('Helvetica', '', 12)

    def header(self):
        self.set_font('Helvetica', 'B', 12)
        header_text = (
            f"Model: {self.config.get('model', 'N/A')} | "
            f"Voters: {self.config.get('voters', 'N/A')} | "
            f"Regime: {self.config.get('few_shot', 'N/A')}-shot | "
            f"Domain: {self.config.get('domain', 'N/A')} | "
            f"Timestamp: {self.config.get('timestamp', 'N/A')}"
        )
        self.cell(0, 10, header_text, 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, content):
        self.set_font('Helvetica', '', 12)
        self.multi_cell(0, 10, content)
        self.ln()

    def generate_content(self, data):
        """Generates all the sections for the PDF report."""
        self.add_agent_level_results(data['agent_level'])
        self.add_polarity_confusion_matrix(data['agent_level']['sentiment'])
        self.add_pipeline_level_acos(data['pipeline_level'])
        self.add_cost_summary(data['cost'])
        self.add_error_breakdown(data['errors'])
        self.add_top_k_category_errors(data['agent_level']['category'])
        self.add_qualitative_examples(data.get('case_studies', []))
    
    def add_agent_level_results(self, agent_level_data):
        self.chapter_title("2. Agent-level results")
        
        headers = ["Metric", "AOE", "Sentiment", "Category"]
        rows = [
            ["Precision", f"{agent_level_data['aoe'].get('precision', 0):.3f}", "N/A", "N/A"],
            ["Recall", f"{agent_level_data['aoe'].get('recall', 0):.3f}", "N/A", "N/A"],
            ["F1-Score", f"{agent_level_data['aoe'].get('f1', 0):.3f}", "N/A", "N/A"],
            ["Accuracy", "N/A", f"{agent_level_data['sentiment'].get('accuracy', 0):.3f}", f"{agent_level_data['category'].get('accuracy', 0):.3f}"],
            ["Macro-F1", "N/A", f"{agent_level_data['sentiment'].get('macro_f1', 0):.3f}", f"{agent_level_data['category'].get('macro_f1', 0):.3f}"],
        ]

        self.set_font('Helvetica', 'B', 10)
        col_widths = [40, 30, 30, 30]
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 10, header, 1, 0, 'C')
        self.ln()
        
        self.set_font('Helvetica', '', 10)
        for row in rows:
            for i, cell_text in enumerate(row):
                self.cell(col_widths[i], 10, str(cell_text), 1, 0, 'C')
            self.ln()
            
        self.ln()
        self.set_font('Helvetica', 'I', 9)
        evaluated_pairs = agent_level_data['sentiment'].get('evaluated_pairs', 'N/A')
        self.multi_cell(0, 5, f"Footnote: Sentiment & Category metrics computed on N = {evaluated_pairs} correctly-extracted pairs.")
        self.ln(10)

    def add_polarity_confusion_matrix(self, sentiment_data):
        self.chapter_title("3. Polarity confusion matrix")
        
        matrix = sentiment_data.get('confusion_matrix', [])
        labels = sentiment_data.get('labels', ['positive', 'neutral', 'negative'])
        
        if not matrix:
            self.chapter_body("No confusion matrix data available.")
            return

        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save the plot to a temporary file
        img_path = "temp_confusion_matrix.png"
        plt.savefig(img_path)
        plt.close()
        
        self.image(img_path, x=self.get_x(), w=100)
        os.remove(img_path) # Clean up the temporary file
        
        self.ln(5)
        self.set_font('Helvetica', 'I', 9)
        total_pairs = sum(sum(row) for row in matrix)
        self.cell(0, 10, f"Matrix over {total_pairs} pairs.", 0, 1, 'C')
        self.ln(10)

    def add_pipeline_level_acos(self, pipeline_data):
        self.chapter_title("4. Pipeline-level ACOS scores")
        headers = ["Metric", "F1-Score"]
        rows = [
            ["Exact-match F1", f"{pipeline_data.get('exact_match_f1', 0):.3f}"],
            ["Partial-match F1", f"{pipeline_data.get('partial_match_f1', 0):.3f}"],
        ]
        
        self.set_font('Helvetica', 'B', 10)
        self.cell(60, 10, headers[0], 1, 0, 'C')
        self.cell(60, 10, headers[1], 1, 0, 'C')
        self.ln()
        
        self.set_font('Helvetica', '', 10)
        for row in rows:
            self.cell(60, 10, row[0], 1, 0, 'L')
            self.cell(60, 10, str(row[1]), 1, 0, 'C')
            self.ln()
        self.ln(10)

    def add_cost_summary(self, cost_data):
        self.chapter_title("5. Cost summary")
        
        # Check if we have detailed token data
        if 'detailed_tokens' in cost_data:
            detailed_tokens = cost_data['detailed_tokens']
            ensemble_size = cost_data.get('ensemble_size', 1)
            
            # Overall summary first
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, "Overall Summary", 0, 1, 'L')
            self.set_font('Helvetica', '', 10)
            
            overall_rows = [
                ("Total cost", f"${cost_data.get('total_cost_usd', 0):.4f}"),
                ("# sentences", cost_data.get('sentences', 'N/A')),
                ("Avg. cost / sentence", f"${cost_data.get('avg_cost_usd_per_sentence', 0):.6f}"),
                ("Avg. tokens / sentence", f"{cost_data.get('avg_tokens_per_sentence', 0):.1f}"),
                ("Ensemble size (unified)", ensemble_size),
            ]

            for title, value in overall_rows:
                self.set_font('Helvetica', 'B', 10)
                self.cell(60, 8, title, 0, 0, 'L')
                self.set_font('Helvetica', '', 10)
                self.cell(0, 8, str(value), 0, 1, 'L')
            
            self.ln(5)
            
            # Detailed per-agent breakdown
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, "Per-Agent Token Breakdown", 0, 1, 'L')
            
            # Table headers
            self.set_font('Helvetica', 'B', 9)
            col_widths = [30, 25, 25, 25, 25, 30]
            headers = ["Agent", "Input", "Output", "Total", "Cost", "Notes"]
            
            for i, header in enumerate(headers):
                self.cell(col_widths[i], 8, header, 1, 0, 'C')
            self.ln()
            
            # Agent rows
            self.set_font('Helvetica', '', 9)
            agents = ["unified", "sentiment", "category"]
            
            for agent in agents:
                if agent in detailed_tokens:
                    tokens = detailed_tokens[agent]
                    input_tokens = tokens.get('input_tokens', 0)
                    output_tokens = tokens.get('output_tokens', 0)
                    total_tokens = tokens.get('total_tokens', 0)
                    cost = tokens.get('cost', 0)
                    
                    # Special note for unified extractor
                    note = f"×{ensemble_size} voters" if agent == "unified" and ensemble_size > 1 else ""
                    
                    self.cell(col_widths[0], 8, agent.capitalize(), 1, 0, 'L')
                    self.cell(col_widths[1], 8, f"{input_tokens:,}", 1, 0, 'R')
                    self.cell(col_widths[2], 8, f"{output_tokens:,}", 1, 0, 'R')
                    self.cell(col_widths[3], 8, f"{total_tokens:,}", 1, 0, 'R')
                    self.cell(col_widths[4], 8, f"${cost:.4f}", 1, 0, 'R')
                    self.cell(col_widths[5], 8, note, 1, 0, 'C')
                    self.ln()
            
            # Totals row
            if 'totals' in detailed_tokens:
                totals = detailed_tokens['totals']
                self.set_font('Helvetica', 'B', 9)
                self.cell(col_widths[0], 8, "TOTAL", 1, 0, 'L')
                self.cell(col_widths[1], 8, f"{totals.get('input_tokens', 0):,}", 1, 0, 'R')
                self.cell(col_widths[2], 8, f"{totals.get('output_tokens', 0):,}", 1, 0, 'R')
                self.cell(col_widths[3], 8, f"{totals.get('total_tokens', 0):,}", 1, 0, 'R')
                self.cell(col_widths[4], 8, f"${totals.get('cost', 0):.4f}", 1, 0, 'R')
                self.cell(col_widths[5], 8, "", 1, 0, 'C')
                self.ln()
        
        else:
            # Fallback to simple display if detailed data not available
            rows = [
                ("Avg. tokens / sentence", cost_data.get('avg_tokens_per_sentence', 'N/A')),
                ("Avg. cost / sentence", f"${cost_data.get('avg_cost_usd_per_sentence', 0):.4f}"),
                ("Total cost", f"${cost_data.get('total_cost_usd', 0):.2f}"),
                ("# sentences", cost_data.get('sentences', 'N/A')),
            ]

            for title, value in rows:
                self.set_font('Helvetica', 'B', 10)
                self.cell(60, 10, title, 0, 0, 'L')
                self.set_font('Helvetica', '', 10)
                self.cell(0, 10, str(value), 0, 1, 'L')
        
        self.ln(10)

    def add_error_breakdown(self, error_data):
        self.chapter_title("6. Error breakdown")
        headers = ["Error Type", "Count"]
        rows = [
            ["Extraction", error_data.get('extraction', 'N/A')],
            ["Sentiment", error_data.get('sentiment', 'N/A')],
            ["Category", error_data.get('category', 'N/A')],
        ]

        self.set_font('Helvetica', 'B', 10)
        self.cell(60, 10, headers[0], 1, 0, 'C')
        self.cell(60, 10, headers[1], 1, 0, 'C')
        self.ln()
        
        self.set_font('Helvetica', '', 10)
        for row in rows:
            self.cell(60, 10, row[0], 1, 0, 'L')
            self.cell(60, 10, str(row[1]), 1, 0, 'C')
            self.ln()
        self.ln(10)

    def add_top_k_category_errors(self, category_data):
        self.chapter_title("7. Top-k category errors")
        
        # Absolute errors
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, "Absolute Error Counts", 0, 1, 'L')
        self.set_font('Helvetica', 'B', 10)
        self.cell(60, 10, "Category", 1, 0, 'C')
        self.cell(60, 10, "Errors", 1, 0, 'C')
        self.ln()
        
        self.set_font('Helvetica', '', 10)
        for item in category_data.get('top_errors_absolute', []):
            self.cell(60, 10, item['category'], 1, 0, 'L')
            self.cell(60, 10, str(item['errors']), 1, 0, 'C')
            self.ln()
        self.ln(5)
        
        # Relative errors
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, "Relative Error Rates (>=10 instances)", 0, 1, 'L')
        self.set_font('Helvetica', 'B', 10)
        self.cell(60, 10, "Category", 1, 0, 'C')
        self.cell(40, 10, "Error Rate", 1, 0, 'C')
        self.cell(40, 10, "Instances", 1, 0, 'C')
        self.ln()
        
        self.set_font('Helvetica', '', 10)
        for item in category_data.get('top_errors_relative', []):
            self.cell(60, 10, item['category'], 1, 0, 'L')
            self.cell(40, 10, f"{item['rate']:.2%}", 1, 0, 'C')
            self.cell(40, 10, str(item['instances']), 1, 0, 'C')
            self.ln()
        self.ln(10)

    def add_qualitative_examples(self, case_studies):
        self.chapter_title("8. Qualitative examples")
        
        if not case_studies:
            self.chapter_body("No qualitative examples provided.")
            return
    
        for i, case in enumerate(case_studies):
            self.set_font('Helvetica', 'B', 12)
            self.cell(0, 10, f"Example {i+1}", 0, 1, 'L')
            
            self.set_font('Helvetica', 'I', 10)
            self.multi_cell(0, 6, f"Sentence: {case['sentence']}")
            
            self.set_font('Helvetica', 'B', 10)
            self.cell(0, 8, "Gold ACOS:", 0, 1, 'L')
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 6, str(case['gold_acos']))
            
            self.set_font('Helvetica', 'B', 10)
            self.cell(0, 8, "Predicted ACOS:", 0, 1, 'L')
            self.set_font('Helvetica', '', 10)
            self.multi_cell(0, 6, str(case['predicted_acos']))

            self.set_font('Helvetica', 'B', 10)
            self.cell(0, 8, f"Failing Agent: {case.get('failing_agent', 'N/A')}", 0, 1, 'L')
            
            self.ln(5) 