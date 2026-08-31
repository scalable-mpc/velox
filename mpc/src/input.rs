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
