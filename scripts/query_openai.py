import pandas as pd
import time
import argparse
from typing import Optional, Dict, List
import logging
import os
from datetime import datetime
from openai import OpenAI


class OpenAICSVProcessor:
    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4",
            batch_size: int = 10,
            delay_between_calls: float = 1.0,
            verbose: bool = False,
            debug: bool = False
    ):
        self.model = model
        self.batch_size = batch_size
        self.delay_between_calls = delay_between_calls
        self.verbose = verbose
        self.debug = debug

        # Configure logging
        log_level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

    def construct_prompt(self, row: pd.Series) -> str:
        """Constructs the prompt for OpenAI based on the row data."""
        prompt = f"""Context: {row['story_setup']}

Prior question: {row['priorQ']}
Speaker's statement: {row['speach']}
Follow-up question: {row['speachQ']}

Based on this context, please evaluate what is known and respond with your assessment.
"""
        return prompt

    def get_openai_response(self, prompt: str) -> Optional[str]:
        """Gets response from OpenAI API with error handling."""
        try:
            self.logger.debug(f"Sending prompt to OpenAI: {prompt}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing pragmatic language use."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            response_text = response.choices[0].message.content.strip()
            self.logger.debug(f"Received response: {response_text}")

            return response_text

        except Exception as e:
            self.logger.error(f"Error getting OpenAI response: {str(e)}")
            return None

    def process_csv(self, input_file: str, output_file: str, start_row: int = 0, end_row: Optional[int] = None):
        """Process the CSV file in batches."""
        try:
            # Read the CSV file
            df = pd.read_csv(input_file)
            self.logger.info(f"Loaded CSV with {len(df)} rows")

            # Set end row if not specified
            if end_row is None:
                end_row = len(df)

            # Validate row range
            start_row = max(0, min(start_row, len(df)))
            end_row = max(0, min(end_row, len(df)))

            # Create responses column if it doesn't exist
            if 'ai_response' not in df.columns:
                df['ai_response'] = None

            # Process in batches
            for i in range(start_row, end_row, self.batch_size):
                batch_end = min(i + self.batch_size, end_row)
                self.logger.info(f"Processing batch from row {i} to {batch_end}")

                for idx in range(i, batch_end):
                    if pd.isna(df.loc[idx, 'ai_response']):
                        prompt = self.construct_prompt(df.loc[idx])
                        response = self.get_openai_response(prompt)

                        if response:
                            df.loc[idx, 'ai_response'] = response
                            self.logger.info(f"Processed row {idx}")
                        else:
                            self.logger.warning(f"Failed to get response for row {idx}")

                        time.sleep(self.delay_between_calls)

                # Save intermediate results
                df.to_csv(output_file, index=False)
                self.logger.info(f"Saved progress to {output_file}")

            return df

        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Process CSV with OpenAI API')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output_file', help='Output CSV file path')
    parser.add_argument('--api_key', help='OpenAI API key', required=True)
    parser.add_argument('--model', default='gpt-4', help='OpenAI model to use')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of rows to process in each batch')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls in seconds')
    parser.add_argument('--start_row', type=int, default=0, help='Starting row number')
    parser.add_argument('--end_row', type=int, help='Ending row number')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Generate default output filename if not provided
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = f"{base_name}_answered_{timestamp}.csv"

    processor = OpenAICSVProcessor(
        api_key=args.api_key,
        model=args.model,
        batch_size=args.batch_size,
        delay_between_calls=args.delay,
        verbose=args.verbose,
        debug=args.debug
    )

    processor.process_csv(
        args.input_file,
        args.output_file,
        args.start_row,
        args.end_row
    )


if __name__ == "__main__":
    main()