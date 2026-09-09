use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::anyhow;
use fields::ProtocolField;
use lambdaworks_math::field::element::FieldElement;

/// Reads first k lines from the primary file, or fallback file if primary doesn't exist
/// Validates that each line fits in one field element (`F::MAX_INPUT_PAYLOAD` bytes)
pub fn read_input_from_files<F: ProtocolField>(
    primary_file_path: &str,
    fallback_file_path: &str,
    k: usize,
) -> Result<Vec<FieldElement<F>>,anyhow::Error> {
    // Check which file to use
    let file_path = if Path::new(primary_file_path).exists() {
        log::info!("Primary file exists, reading from: {}", primary_file_path);
        primary_file_path
    } else {
        log::info!("Primary file doesn't exist, reading from fallback: {}", fallback_file_path);
        fallback_file_path
    };

    // Open and read the file
    let file = File::open(file_path)
        .map_err(|e| anyhow!("Failed to open file {}: {}", file_path, e))?;
    
    let reader = BufReader::new(file);
    let mut converted_fes = Vec::new();
    let mut line_count = 0;

    for line_result in reader.lines() {
        if line_count >= k {
            break;
        }

        let line = line_result
            .map_err(|e| anyhow!("Failed to read line {}: {}", line_count + 1, e))?;
        
        // Validate that the line can be represented as a 250-bit binary string
        let conversion_output = F::encode_ascii(&line);
        if conversion_output.is_none() {
            return Err(anyhow!(
                "Line {} is {} bytes, more than the {} one field element holds: '{}'",
                line_count + 1,
                line.len(),
                max_input_payload::<F>(),
                line
            ));
        }

        converted_fes.push(conversion_output.unwrap());
        line_count += 1;
    }

    if line_count < k {
        log::error!("File {} contains only {} inputs, but {} inputs were requested", 
                    file_path, line_count, k);
        return Err(anyhow!("Insufficient inputs in file {}", file_path));
    }

    log::info!("Successfully read {} lines from {}", converted_fes.len(), file_path);
    Ok(converted_fes)
}

/// Longest input line that fits in one element, for the error message below.
/// The packing itself is `F::encode_ascii` — it depends on the field's byte
/// layout, so it lives with the field rather than here.
fn max_input_payload<F: ProtocolField>() -> usize {
    F::MAX_INPUT_PAYLOAD
}

/// Reads the first `k` numeric inputs from the primary file, or the fallback
/// file if the primary does not exist.
///
/// This is the arithmetic-circuit counterpart of
/// [`read_input_from_files`]: an arithmetic circuit's input wires carry field
/// elements, not text, so the lines are decimal integers rather than the ASCII
/// payloads `F::encode_ascii` packs. Blank lines and `#` comments are skipped,
/// so an input file can be annotated the way the circuit files are.
///
/// Values are reduced modulo the field's order, and may be written larger than
/// a `u64` — the digits are folded in one at a time, which works at any field
/// size.
pub fn read_numeric_input_from_files<F: ProtocolField>(
    primary_file_path: &str,
    fallback_file_path: &str,
    k: usize,
) -> Result<Vec<FieldElement<F>>, anyhow::Error> {
    let file_path = if Path::new(primary_file_path).exists() {
        log::info!("Primary file exists, reading from: {}", primary_file_path);
        primary_file_path
    } else {
        log::info!("Primary file doesn't exist, reading from fallback: {}", fallback_file_path);
        fallback_file_path
    };

    let file = File::open(file_path)
        .map_err(|e| anyhow!("Failed to open file {}: {}", file_path, e))?;

    let reader = BufReader::new(file);
    let mut values = Vec::with_capacity(k);

    for (index, line_result) in reader.lines().enumerate() {
        if values.len() >= k {
            break;
        }
        let line = line_result
            .map_err(|e| anyhow!("Failed to read line {} of {}: {}", index + 1, file_path, e))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        values.push(parse_decimal::<F>(line).ok_or_else(|| {
            anyhow!("Line {} of {} is not a decimal integer: '{}'", index + 1, file_path, line)
        })?);
    }

    if values.len() < k {
        log::error!("File {} contains only {} inputs, but {} inputs were requested",
                    file_path, values.len(), k);
        return Err(anyhow!("Insufficient inputs in file {}", file_path));
    }

    log::info!("Successfully read {} numeric inputs from {}", values.len(), file_path);
    Ok(values)
}

/// Folds a decimal string into a field element, digit by digit, so a value
/// wider than a `u64` still lands correctly in the larger fields.
fn parse_decimal<F: ProtocolField>(text: &str) -> Option<FieldElement<F>> {
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }

    let ten = FieldElement::<F>::from(10u64);
    let mut value = FieldElement::<F>::zero();
    for digit in digits.bytes() {
        value = value * ten.clone() + FieldElement::<F>::from((digit - b'0') as u64);
    }
    Some(if negative { -value } else { value })
}
