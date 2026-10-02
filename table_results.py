import sys
import argparse
import re

def highlight_markdown_table(md_table):
    lines = md_table.strip().split('\n')
    
    parsed_data = []
    for line in lines:
        if '|' in line:
            parsed_data.append(line.split('|'))
        else:
            parsed_data.append(line)
            
    table_row_indices = [i for i, r in enumerate(parsed_data) if isinstance(r, list)]
    
    if len(table_row_indices) < 3:
        return "Error: A valid Markdown table (with header and separator) was not found."
        
    data_row_indices = table_row_indices[2:]
    num_cols = len(parsed_data[table_row_indices[0]])
    
    for col_idx in range(num_cols):
        col_vals = []
        
        for row_idx in data_row_indices:
            cells = parsed_data[row_idx]
            if col_idx < len(cells):
                cell_text = cells[col_idx].strip()
                if not cell_text:
                    continue
                    
                match = re.search(r'[-+]?\d*\.?\d+', cell_text.replace(',', ''))
                if match:
                    try:
                        val = float(match.group())
                        col_vals.append((val, row_idx))
                    except ValueError:
                        pass
                        
        if not col_vals:
            continue
            
        unique_vals = sorted(list(set(v for v, idx in col_vals)), reverse=True)
        
        if not unique_vals:
            continue
            
        highest = unique_vals[0]
        second_highest = unique_vals[1] if len(unique_vals) > 1 else None
        
        for val, row_idx in col_vals:
            orig_cell = parsed_data[row_idx][col_idx]
            stripped_cell = orig_cell.strip()
            
            clean_cell = re.sub(r'^\*\*(.*?)\*\*$', r'\1', stripped_cell)
            clean_cell = re.sub(r'^<u>(.*?)</u>$', r'\1', clean_cell)
            
            if val == highest:
                parsed_data[row_idx][col_idx] = f" **{clean_cell}** "
            elif second_highest is not None and val == second_highest:
                parsed_data[row_idx][col_idx] = f" <u>{clean_cell}</u> "
                
    result_lines = []
    for item in parsed_data:
        if isinstance(item, list):
            result_lines.append("|".join(item))
        else:
            result_lines.append(item)
            
    return "\n".join(result_lines)


def main():
    parser = argparse.ArgumentParser(description="Highlight highest (bold) and second highest (underline) values in a Markdown table.")
    parser.add_argument(
        "filepath", 
        nargs="?", 
        help="Path to the markdown file. If omitted, you can paste the table directly into the terminal."
    )
    
    args = parser.parse_args()
    
    if args.filepath:
        try:
            with open(args.filepath, 'r', encoding='utf-8') as f:
                input_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Paste your Markdown table below.")
        print("(Press Ctrl+D on Linux/Mac or Ctrl+Z on Windows, then Enter, to finish):")
        print("-" * 50)
        input_text = sys.stdin.read()
        
    if not input_text.strip():
        print("Error: No input provided.", file=sys.stderr)
        sys.exit(1)
        
    formatted_table = highlight_markdown_table(input_text)
    
    if not args.filepath:
        print("\n" + "=" * 50)
        print("FORMATTED TABLE:")
        print("=" * 50 + "\n")
        
    print(formatted_table)

if __name__ == "__main__":
    main()