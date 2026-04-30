namespace WeatherAiDotNet.Models;

/// <summary>
/// Represents the outcome of validating an external command-line tool at startup.
/// Used by <see cref="Services.CliToolService"/> to report whether an executable
/// (e.g., the OCR binary) could be successfully launched and returned a zero exit code.
/// </summary>
/// <param name="Success"><see langword="true"/> if the tool launched successfully; otherwise <see langword="false"/>.</param>
/// <param name="Message">
/// An empty string on success, or a human-readable error/output message that explains
/// why the validation failed.
/// </param>
public record ToolCheckResult(bool Success, string Message);