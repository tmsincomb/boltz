#!/usr/bin/env python3
"""
Convert ProteinMPNN list.csv and PDB files to Boltz YAML format.

Usage:
    python rcsb2boltz.py --seq-file /path/to/list.csv --pdb-folder /path/to/pdb/
"""

import argparse
import csv
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict


def parse_chainid(chainid: str) -> tuple[str, str]:
    """Parse CHAINID into PDB code and chain.
    
    Args:
        chainid: Format like "6y2d_A" 
        
    Returns:
        Tuple of (pdb_code, chain)
    """
    parts = chainid.split('_')
    if len(parts) >= 2:
        pdb_code = parts[0].lower()
        chain = '_'.join(parts[1:])  # Handle cases like "1abc_A_B"
        return pdb_code, chain
    return None, None


def find_pdb_file(pdb_code: str, pdb_folder: Path) -> Optional[Path]:
    """Find PDB file in the folder structure.
    
    Args:
        pdb_code: PDB code like "6y2d"
        pdb_folder: Root PDB folder path
        
    Returns:
        Path to PDB file if found, None otherwise
    """
    # Try common subdirectory patterns
    patterns = [
        f"{pdb_code[:2]}/{pdb_code}.pdb1",  # e.g., y2/6y2d.pdb1
        f"{pdb_code[1:3]}/{pdb_code}.pdb1",  # e.g., 2d/6y2d.pdb1
        f"**/{pdb_code}.pdb1",  # Search recursively
        f"**/{pdb_code}.pdb",   # Try .pdb extension
    ]
    
    for pattern in patterns:
        matches = list(pdb_folder.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Convert ProteinMPNN list.csv and PDB files to Boltz YAML format'
    )
    parser.add_argument(
        '--seq-file', 
        required=True,
        help='Path to list.csv file with CHAINID and SEQUENCE columns'
    )
    parser.add_argument(
        '--pdb-folder',
        required=True,
        help='Path to folder containing PDB files (e.g., pdb_2025sep11)'
    )
    parser.add_argument(
        '--output',
        default='boltz.yaml',
        help='Output YAML file path (default: boltz.yaml)'
    )
    parser.add_argument(
        '--filter-pdb',
        nargs='+',
        help='Only include specific PDB codes (e.g., 6y2d 7y2p)'
    )
    parser.add_argument(
        '--filter-chain',
        nargs='+',
        help='Only include specific CHAINIDs (e.g., 6y2d_A 7y2p_B)'
    )
    parser.add_argument(
        '--max-chains',
        type=int,
        help='Maximum number of chains to include (default: all)'
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show progress while processing large files'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Process in batches and create multiple output files (e.g., 1000 chains per file)'
    )
    parser.add_argument(
        '--skip-missing-templates',
        action='store_true',
        help='Skip chains without found PDB templates instead of erroring'
    )
    parser.add_argument(
        '--no-group-sequences',
        action='store_true',
        help='Do not group chains with identical sequences (treat each independently)'
    )
    
    args = parser.parse_args()
    
    # Parse paths
    csv_path = Path(args.seq_file)
    pdb_folder = Path(args.pdb_folder)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1
    
    if not pdb_folder.exists():
        print(f"Error: PDB folder not found: {pdb_folder}")
        return 1
    
    # Read CSV file
    chains_data = []
    total_rows = 0
    skipped_x = 0
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            chainid = row.get('CHAINID', '')
            sequence = row.get('SEQUENCE', '')
            
            # Skip invalid entries
            if not chainid or not sequence:
                continue
            
            # Parse chain ID early for filtering
            pdb_code, chain = parse_chainid(chainid)
            if not pdb_code:
                continue
                
            # Apply PDB filter early
            if args.filter_pdb and pdb_code not in args.filter_pdb:
                continue
            
            # Apply chain filter
            if args.filter_chain and chainid not in args.filter_chain:
                continue
            
            # Skip sequences with XXX (missing residues)
            if 'X' in sequence:
                skipped_x += 1
                if not args.progress or skipped_x <= 10:
                    print(f"Skipping {chainid}: contains missing residues (X)")
                continue
            
            # Show progress
            if args.progress and row_num % 10000 == 0:
                print(f"Processed {row_num:,} rows, found {len(chains_data):,} valid chains...")
            
            chains_data.append({
                'chainid': chainid,
                'pdb_code': pdb_code,
                'chain': chain,
                'sequence': sequence
            })
            
            # Apply max chains limit during reading
            if args.max_chains and len(chains_data) >= args.max_chains:
                break
    
    if not chains_data:
        print("No valid chains found after filtering")
        return 1
    
    # Max chains limit already applied during reading
    
    if args.progress and skipped_x > 10:
        print(f"... and {skipped_x - 10:,} more chains with missing residues")
    
    print(f"\nProcessing {len(chains_data):,} valid chains...")
    if skipped_x > 0:
        print(f"  (Skipped {skipped_x:,} chains with missing residues)")
    
    # Build YAML structure
    yaml_data = {
        'version': 1,
        'sequences': [],
        'templates': []
    }
    
    # Group chains by sequence to detect identical sequences
    seq_to_chains = defaultdict(list)
    for chain in chains_data:
        seq_to_chains[chain['sequence']].append(chain)
    
    # Create sequence entries
    used_pdb_files = set()
    
    if args.no_group_sequences:
        # Treat each chain independently
        for chain in chains_data:
            yaml_data['sequences'].append({
                'protein': {
                    'id': chain['chainid'],
                    'sequence': chain['sequence']
                }
            })
    else:
        # Group by sequence (default behavior)
        for sequence, chain_group in seq_to_chains.items():
            if len(chain_group) == 1:
                # Single chain with this sequence
                chain = chain_group[0]
                yaml_data['sequences'].append({
                    'protein': {
                        'id': chain['chainid'],
                        'sequence': sequence
                    }
                })
            else:
                # Multiple chains with identical sequence
                chain_ids = [c['chainid'] for c in chain_group]
                yaml_data['sequences'].append({
                    'protein': {
                        'id': chain_ids,
                        'sequence': sequence
                    }
                })
        
    # Track PDB files needed
    for chain in chains_data:
        pdb_file = find_pdb_file(chain['pdb_code'], pdb_folder)
        if pdb_file:
            used_pdb_files.add((chain['chainid'], chain['pdb_code'], pdb_file))
        elif not args.skip_missing_templates:
            print(f"Error: PDB file not found for {chain['pdb_code']}")
            return 1
        else:
            print(f"Warning: PDB file not found for {chain['pdb_code']}, skipping template")
    
    # Create template entries
    pdb_to_chains = defaultdict(list)
    for chainid, pdb_code, pdb_file in used_pdb_files:
        pdb_to_chains[str(pdb_file)].append((chainid, pdb_code))
    
    for pdb_file, chain_list in pdb_to_chains.items():
        if len(chain_list) == 1:
            # Single chain from this PDB
            chainid, pdb_code = chain_list[0]
            _, chain_letter = parse_chainid(chainid)
            
            template_entry = {
                'pdb': str(pdb_file),
                'chain_id': chainid,
                'template_id': f"{chain_letter}1"  # PDB chain naming convention
            }
        else:
            # Multiple chains from same PDB
            chain_ids = [c[0] for c in chain_list]
            template_ids = []
            for chainid, _ in chain_list:
                _, chain_letter = parse_chainid(chainid)
                template_ids.append(f"{chain_letter}1")
            
            template_entry = {
                'pdb': str(pdb_file),
                'chain_id': chain_ids,
                'template_id': template_ids
            }
        
        yaml_data['templates'].append(template_entry)
    
    # Write YAML file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nSuccessfully created {output_path}")
    print(f"  - {len(yaml_data['sequences'])} unique sequences")
    print(f"  - {len(yaml_data['templates'])} template entries")
    print(f"  - {len(used_pdb_files)} total chain-template mappings")
    
    # Print example command
    print(f"\nTo run Boltz with this file:")
    print(f"  boltz predict {output_path} --use_msa_server")
    
    return 0


if __name__ == '__main__':
    exit(main())