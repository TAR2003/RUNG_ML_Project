# Log Format Compatibility Update

## Overview
The `plot_logs.py` script has been enhanced to support both **old and new log file formats** seamlessly, ensuring backward compatibility while supporting the latest format changes.

## Changes Made

### 1. Enhanced Regex Patterns (Lines 52-61)
- **Old regexes preserved**: `_RE_BUDGET`, `_RE_CLEAN_SUM`, `_RE_ATK_SUM`, `_RE_CLEAN_DONE`
- **New regexes added**: `_RE_CLEAN_LINE`, `_RE_ATK_LINE` for legacy format detection
  - These detect "Clean:" and "Attacked:" summary lines that may appear without `Budget:` sections

### 2. Improved `parse_attack_log()` Function (Lines 103-189)
**Dual-format support:**
- **First pass**: Tries new format with `Budget:` sections (structure: one budget per section with Clean/Attacked summaries)
- **Fallback pass**: If new format yields no results, tries legacy format (consecutive Clean:/Attacked: lines without Budget prefix)
- **Synthetic budgets**: Legacy format files without explicit budget values are assigned synthetic budget values (0.0, 0.1, 0.2, ...) for consistent data representation

### 3. Enhanced `parse_clean_log()` Function (Lines 191-219)
**Multi-format support:**
1. Tries standard clean log format first: `"model … done, clean acc: X±Y"` pattern
2. Falls back to alternative format: `"Clean: X±Y"` summary lines (useful if file ends prematurely or has different structure)

### 4. New Format Detection Function `_detect_log_type()` (Lines 222-249)
**Intelligent format detection:**
- Analyzes file content to determine actual format (attack, clean, or unknown)
- Returns: `"attack"`, `"clean"`, or `"unknown"`
- Checks for key markers:
  - `Budget:` + `Attacked:` → Attack format (new)
  - `Attacked:` (without Budget) → Attack format (legacy)
  - `model ... done, clean acc:` → Clean format

### 5. Robust File Discovery (Lines 263-327)
**Smart file handling:**
- Uses format detection to correctly classify files regardless of directory location
- **Attack files in attack/ directory**: Parsed as attack logs with appropriate labeling
- **Clean files in clean/ directory**: Parsed as clean logs
- **Mixed files** (e.g., attack data in clean/): Correctly reclassified and processed in the appropriate data structure
- Provides informative messages when files are found in unexpected locations

## Supported Log Formats

### New Format (Current)
```
Budget: 0.05
...individual results...
Clean: 0.8064±0.0121: [list of accuracies]
Attacked: 0.7809±0.0321: [list of accuracies]

Budget: 0.1
...
```

### Old Format (Legacy)
```
Fit RUNG_parametric_gamma | lr=0.05, ...
Epoch 50 | loss=0.0128 | val=0.7390 | ...
...
Acc: [0.8153923153877258]
...
model RUNG_parametric_gamma done, clean acc: 0.8064±0.0121
```

### Mixed/Alternative Format
```
Clean: 0.8064±0.0121: [list]
Attacked: 0.7809±0.0321: [list]
(Without Budget: sections)
```

## Benefits

1. **Backward Compatibility**: Existing old-format logs continue to work without modification
2. **Forward Compatibility**: New format logs are automatically detected and processed correctly
3. **Flexible File Organization**: Files don't have to be in "correct" directories—content detection handles misplaced files
4. **Graceful Degradation**: Unknown formats are skipped with informative messages rather than causing crashes
5. **Zero User Action Required**: No manual conversion or reorganization needed when log formats change

## Testing

Run the script to verify:
```bash
python plot_logs.py
```

The script will:
- Automatically detect all log file formats
- Generate appropriate figures for both attack and clean data
- Report file detection results in the console output
- Skip or relocate files as needed based on detected format

## Error Handling

- Files with unknown formats are skipped with a message: `"→  unknown format (skipped)"`
- Files in unexpected directories are correctly reclassified by content analysis
- Unicode normalization handles special characters (±, −) in different encodings
- Parse failures don't crash the entire script—processing continues with remaining files
