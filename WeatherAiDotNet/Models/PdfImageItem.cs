namespace WeatherAiDotNet.Models;

/// <summary>
/// Represents an image that was extracted from a PDF page.
/// If OCR is configured, the image is passed through the OCR tool and the resulting
/// text is stored in <see cref="IndexText"/> so images containing diagrams or
/// captions can be searched semantically, just like regular text chunks.
/// </summary>
/// <param name="RelativePath">
/// Path to the saved PNG file on disk, relative to the current working directory.
/// </param>
/// <param name="PageNumber">One-based page number within the source PDF.</param>
/// <param name="ImageNumber">One-based position of the image on its page.</param>
/// <param name="IndexText">
/// Text that will be embedded and stored in the vector database. Contains metadata
/// tags plus any OCR text extracted from the image, or a placeholder if no OCR text
/// was available.
/// </param>
public record PdfImageItem(string RelativePath, int PageNumber, int ImageNumber, string IndexText);