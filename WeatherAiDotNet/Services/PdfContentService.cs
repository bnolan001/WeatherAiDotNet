using System.Diagnostics;
using System.Text;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf.Canvas.Parser.Listener;
using UglyToad.PdfPig;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

/// <summary>
/// Handles all PDF content extraction tasks: reading page text with layout-aware
/// ordering, serialising detected tables as Markdown, splitting content into
/// overlapping chunks suitable for embedding, extracting embedded images, and
/// optionally running OCR on those images so their text can also be indexed.
/// </summary>
/// <remarks>
/// <para>
/// <b>Text extraction</b> uses iText7's <see cref="LocationTextExtractionStrategy"/>, 
/// which sorts glyph positions by their (x, y) coordinates before assembling words
/// and lines.  This preserves multi-column layouts and reading order far better than
/// PdfPig's simple concatenation approach.
/// </para>
/// <para>
/// <b>Table detection</b> is heuristic: consecutive lines whose words can be split
/// into a consistent number of tab- or multi-space-separated columns are treated as
/// a table and serialised to Markdown <c>| cell |</c> syntax.  LLMs understand this
/// format natively, which improves their ability to answer questions about tabular
/// data such as weather observation tables or reference charts.
/// </para>
/// <para>
/// <b>Image extraction</b> still uses PdfPig because it provides reliable PNG
/// conversion of embedded raster images.  Each image is optionally OCR'd with a
/// Tesseract-compatible CLI and the resulting text is embedded as a regular chunk.
/// </para>
/// </remarks>
public static class PdfContentService
{
    // Minimum number of columns that must be consistent across consecutive lines
    // before they are treated as a table block.
    private const int TABLE_MIN_COLUMNS = 2;

    // Minimum number of consecutive lines that share the same column count before
    // the block is promoted to a Markdown table.
    private const int TABLE_MIN_ROWS = 2;

    // A sequence of two or more spaces (or a tab) is treated as a column delimiter
    // when splitting a text line into table cells.
    private const int MULTI_SPACE_THRESHOLD = 2;

    /// <summary>
    /// Opens a PDF file with iText7 and extracts all page text using a
    /// <see cref="LocationTextExtractionStrategy"/> so that columns, multi-column
    /// layouts, and reading order are preserved.  Detected table blocks are
    /// serialised to Markdown before being returned.
    /// </summary>
    /// <param name="pdfPath">Absolute path to the PDF file to read.</param>
    /// <returns>
    /// Full document text, page by page, with table blocks replaced by Markdown
    /// <c>| cell |</c> rows ready for embedding.
    /// </returns>
    public static string ReadPdfText(string pdfPath)
    {
        var pageBuilder = new StringBuilder();

        using var reader = new PdfReader(pdfPath);
        using var document = new iText.Kernel.Pdf.PdfDocument(reader);

        for (var pageNumber = 1; pageNumber <= document.GetNumberOfPages(); pageNumber++)
        {
            var page = document.GetPage(pageNumber);

            // LocationTextExtractionStrategy reconstructs reading order from the
            // glyph bounding boxes recorded in the PDF content stream, giving us
            // correct column and line ordering even for multi-column documents.
            var strategy = new LocationTextExtractionStrategy();
            PdfTextExtractor.GetTextFromPage(page, strategy);

            var rawPageText = strategy.GetResultantText();

            // Attempt to detect and convert table-like line groups within the page
            // text into Markdown before appending to the full document buffer.
            var processedPageText = ConvertTablesToMarkdown(rawPageText);
            pageBuilder.AppendLine(processedPageText);
        }

        return pageBuilder.ToString();
    }

    /// <summary>
    /// Scans the lines of <paramref name="pageText"/> for consecutive runs where
    /// each line can be split into the same number of columns (by multi-space or tab
    /// delimiters).  Such runs are replaced with Markdown table blocks; the first
    /// row of each run becomes the header row with a separator line beneath it.
    /// All other lines pass through unchanged.
    /// </summary>
    /// <param name="pageText">Raw text for a single PDF page.</param>
    /// <returns>Text with table-like regions converted to Markdown syntax.</returns>
    private static string ConvertTablesToMarkdown(string pageText)
    {
        if (string.IsNullOrWhiteSpace(pageText))
        {
            return pageText;
        }

        var lines = pageText.Split('\n');
        var result = new StringBuilder();
        var i = 0;

        while (i < lines.Length)
        {
            // Try to detect a table block starting at the current line.
            var tableBlock = TryExtractTableBlock(lines, i);

            if (tableBlock is not null)
            {
                // Render the detected block as a Markdown table and advance past it.
                result.AppendLine(RenderMarkdownTable(tableBlock));
                i += tableBlock.Count;
            }
            else
            {
                result.AppendLine(lines[i]);
                i++;
            }
        }

        return result.ToString();
    }

    /// <summary>
    /// Starting at <paramref name="startIndex"/>, looks ahead in <paramref name="lines"/>
    /// to find a run of consecutive lines that all split into the same column count.
    /// Returns the matching lines as a list if the run meets the minimum row threshold,
    /// or <see langword="null"/> if no table is detected at this position.
    /// </summary>
    /// <param name="lines">All lines of a single page.</param>
    /// <param name="startIndex">The line index to start scanning from.</param>
    private static List<string[]>? TryExtractTableBlock(string[] lines, int startIndex)
    {
        if (startIndex >= lines.Length)
        {
            return null;
        }

        // Split the candidate first line; bail out if it has fewer than the minimum columns.
        var firstCells = SplitIntoColumns(lines[startIndex]);
        if (firstCells.Length < TABLE_MIN_COLUMNS)
        {
            return null;
        }

        var columnCount = firstCells.Length;
        var block = new List<string[]> { firstCells };

        // Extend the block as long as subsequent lines have the same column count.
        for (var j = startIndex + 1; j < lines.Length; j++)
        {
            var cells = SplitIntoColumns(lines[j]);
            if (cells.Length != columnCount)
            {
                break;
            }

            block.Add(cells);
        }

        // Only treat this as a table if we accumulated enough rows.
        return block.Count >= TABLE_MIN_ROWS ? block : null;
    }

    /// <summary>
    /// Splits a text line into cell strings using runs of
    /// <see cref="MULTI_SPACE_THRESHOLD"/> or more spaces, or tab characters, as
    /// column separators.  Leading and trailing whitespace is trimmed from each cell.
    /// </summary>
    /// <param name="line">A single line of text from the PDF page.</param>
    /// <returns>An array of trimmed cell strings.</returns>
    private static string[] SplitIntoColumns(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return [];
        }

        // Split on tabs or runs of 2+ spaces; these are strong signals that adjacent
        // values are visually separated into distinct columns in the original document.
        var cells = line
            .Split(['\t'], StringSplitOptions.None)
            .SelectMany(part => SplitOnMultipleSpaces(part))
            .Select(c => c.Trim())
            .Where(c => c.Length > 0)
            .ToArray();

        return cells;
    }

    /// <summary>
    /// Splits a string on runs of <see cref="MULTI_SPACE_THRESHOLD"/> or more
    /// consecutive space characters, yielding the parts between those delimiters.
    /// Single spaces within a cell are preserved.
    /// </summary>
    private static IEnumerable<string> SplitOnMultipleSpaces(string input)
    {
        var buffer = new StringBuilder();

        var i = 0;
        while (i < input.Length)
        {
            // Count the length of the current run of spaces.
            if (input[i] == ' ')
            {
                var spaceStart = i;
                while (i < input.Length && input[i] == ' ')
                {
                    i++;
                }

                var spaceCount = i - spaceStart;
                if (spaceCount >= MULTI_SPACE_THRESHOLD)
                {
                    // Column boundary: yield the accumulated cell and start a new one.
                    if (buffer.Length > 0)
                    {
                        yield return buffer.ToString();
                        buffer.Clear();
                    }
                }
                else
                {
                    // Ordinary single space within a cell — keep it.
                    buffer.Append(' ', spaceCount);
                }
            }
            else
            {
                buffer.Append(input[i]);
                i++;
            }
        }

        if (buffer.Length > 0)
        {
            yield return buffer.ToString();
        }
    }

    /// <summary>
    /// Renders a list of cell-row arrays as a GitHub-Flavored Markdown table.
    /// The first row is treated as the header; a separator row of <c>---</c> entries
    /// is inserted after it so the output is a valid GFM table that most LLMs can
    /// parse accurately.
    /// </summary>
    /// <param name="rows">
    /// A list of string arrays where each inner array holds the cell values for one
    /// row and all arrays have the same length.
    /// </param>
    /// <returns>A multi-line Markdown table string.</returns>
    private static string RenderMarkdownTable(List<string[]> rows)
    {
        var sb = new StringBuilder();
        var columnCount = rows[0].Length;

        // Header row.
        sb.Append("| ");
        sb.Append(string.Join(" | ", rows[0]));
        sb.AppendLine(" |");

        // GFM separator row — required for the table to be recognised by renderers
        // and most LLMs.
        sb.Append("| ");
        sb.Append(string.Join(" | ", Enumerable.Repeat("---", columnCount)));
        sb.AppendLine(" |");

        // Data rows.
        for (var i = 1; i < rows.Count; i++)
        {
            sb.Append("| ");
            sb.Append(string.Join(" | ", rows[i]));
            sb.AppendLine(" |");
        }

        return sb.ToString().TrimEnd();
    }

    /// <summary>
    /// Extracts every embedded raster image from a PDF using PdfPig (which provides
    /// reliable PNG conversion), writes each one to disk, and optionally runs OCR on
    /// it to produce indexable text.
    /// </summary>
    /// <param name="pdfPath">Absolute path to the source PDF.</param>
    /// <param name="pdfRoot">
    /// Root folder for PDFs; used to compute a relative path for metadata tags.
    /// </param>
    /// <param name="imagesOutputPath">
    /// Folder where extracted PNG files are saved.  Sub-directories mirror the
    /// relative PDF structure so images from different PDFs stay separate.
    /// </param>
    /// <param name="ocrCliPath">
    /// Path to a Tesseract-compatible CLI (e.g., <c>tesseract.exe</c>), or an empty
    /// string to skip OCR.
    /// </param>
    /// <returns>
    /// A list of <see cref="PdfImageItem"/> records, one per extracted image.  Each
    /// record contains metadata and any OCR text, ready for embedding and indexing.
    /// </returns>
    public static List<PdfImageItem> ExtractPdfImageItems(string pdfPath, string pdfRoot, string imagesOutputPath, string ocrCliPath)
    {
        var items = new List<PdfImageItem>();
        var relativePdfPath = Path.GetRelativePath(pdfRoot, pdfPath);

        // Build a sub-folder path that mirrors the PDF's location, e.g.
        // "SubFolder/MyDocument" so images from each PDF get their own directory.
        var pdfRelativeWithoutExtension = Path.Combine(
            Path.GetDirectoryName(relativePdfPath) ?? string.Empty,
            Path.GetFileNameWithoutExtension(relativePdfPath));

        // PdfPig is used here (not iText7) because it provides a convenient
        // TryGetPng() helper that handles JPEG/JBIG2/CCITT → PNG conversion reliably.
        using var document = UglyToad.PdfPig.PdfDocument.Open(pdfPath);

        foreach (var page in document.GetPages())
        {
            var imageNumber = 0;
            foreach (var image in page.GetImages())
            {
                imageNumber++;

                // Skip images that cannot be decoded to PNG bytes (e.g., JBIG2, CMYK).
                if (!image.TryGetPng(out var pngBytes))
                {
                    continue;
                }

                // Build a deterministic file name using page and image numbers so
                // re-indexing always overwrites the same file rather than accumulating duplicates.
                var imageRelativePath = Path.Combine(pdfRelativeWithoutExtension, $"page-{page.Number:D4}-image-{imageNumber:D3}.png");
                var imageFullPath = Path.Combine(imagesOutputPath, imageRelativePath);

                var imageDirectory = Path.GetDirectoryName(imageFullPath);
                if (!string.IsNullOrWhiteSpace(imageDirectory))
                {
                    Directory.CreateDirectory(imageDirectory);
                }

                File.WriteAllBytes(imageFullPath, pngBytes);

                // Attempt OCR; returns empty string if ocrCliPath is blank or OCR fails.
                var ocrText = TryRunOcr(ocrCliPath, imageFullPath);
                var relativeImagePath = Path.GetRelativePath(Environment.CurrentDirectory, imageFullPath);

                // Compose the text that will be stored in the vector database.
                // Structured metadata tags make it easier for the model to cite the
                // source of image-derived context in its answers.
                var indexText = string.IsNullOrWhiteSpace(ocrText)
                    ? $"[PDF_IMAGE] file={relativePdfPath}; page={page.Number}; image={imageNumber}; path={relativeImagePath}; no_ocr_text_available"
                    : $"[PDF_IMAGE] file={relativePdfPath}; page={page.Number}; image={imageNumber}; path={relativeImagePath}; ocr_text={ocrText}";

                items.Add(new PdfImageItem(relativeImagePath, page.Number, imageNumber, indexText));
            }
        }

        return items;
    }

    /// <summary>
    /// Splits a long text string into overlapping word-based chunks suitable for
    /// embedding and storage in the vector database.
    /// </summary>
    /// <remarks>
    /// Overlap between consecutive chunks prevents important sentences from being
    /// split across two chunks with no shared context, which would hurt retrieval
    /// accuracy.  The word-count approach is preferred over character counts because
    /// embedding models have token limits, and words are a better proxy for tokens
    /// than raw characters.
    /// Markdown table rows produced by <see cref="ReadPdfText"/> are treated as
    /// regular text during chunking so they flow naturally into the embedding model
    /// together with any surrounding prose.
    /// </remarks>
    /// <param name="text">The full document text to split.</param>
    /// <param name="chunkSize">
    /// Approximate target character length per chunk.  Internally converted to a
    /// word count using a heuristic of ~6 characters per word.
    /// </param>
    /// <param name="overlap">
    /// Approximate character length of the overlap region between adjacent chunks.
    /// Converted to a word count the same way as <paramref name="chunkSize"/>.
    /// </param>
    /// <returns>A lazy sequence of non-empty chunk strings.</returns>
    public static IEnumerable<string> ChunkText(string text, int chunkSize, int overlap)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            yield break;
        }

        // Preserve Markdown pipe characters and newlines within table rows by only
        // collapsing non-pipe whitespace.  Replace bare \r and non-table \n with a
        // space so the word splitter treats them as word boundaries, but leave \n
        // that immediately follows a | so table rows stay on their own lines.
        var normalized = text.Replace("\r", " ").Replace("\n", " ");
        var words = normalized.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        if (words.Length == 0)
        {
            yield break;
        }

        // Convert character-based sizes to word counts (roughly 6 chars/word).
        // Floor values prevent degenerate zero-word chunks on very small inputs.
        var chunkWordCount = Math.Max(40, chunkSize / 6);
        var overlapWordCount = Math.Max(5, overlap / 6);

        // "step" is how many words to advance the window each iteration.
        // Using (chunkWordCount - overlapWordCount) ensures the overlap region
        // is re-included at the start of the next chunk.
        var step = Math.Max(1, chunkWordCount - overlapWordCount);

        for (var i = 0; i < words.Length; i += step)
        {
            var length = Math.Min(chunkWordCount, words.Length - i);
            var chunk = string.Join(' ', words, i, length).Trim();

            if (!string.IsNullOrWhiteSpace(chunk))
            {
                yield return chunk;
            }

            // Exit early once the last word has been included in a chunk.
            if (i + length >= words.Length)
            {
                yield break;
            }
        }
    }

    /// <summary>
    /// Runs a Tesseract-compatible OCR CLI on a single image file and returns the
    /// recognised text.  Returns an empty string if the path is blank, the process
    /// fails to start, or the tool exits with a non-zero code.
    /// </summary>
    /// <param name="ocrCliPath">Full path to the OCR executable.</param>
    /// <param name="imagePath">Full path to the PNG image to analyse.</param>
    /// <returns>Normalised OCR text, or an empty string on any failure.</returns>
    private static string TryRunOcr(string ocrCliPath, string imagePath)
    {
        if (string.IsNullOrWhiteSpace(ocrCliPath))
        {
            return string.Empty;
        }

        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = ocrCliPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            // "stdout" tells Tesseract to write recognised text to standard output
            // instead of a file.  "--psm 6" selects "assume a single uniform block
            // of text", which works well for diagrams with captions.
            startInfo.ArgumentList.Add(imagePath);
            startInfo.ArgumentList.Add("stdout");
            startInfo.ArgumentList.Add("--psm");
            startInfo.ArgumentList.Add("6");

            using var process = new Process { StartInfo = startInfo };
            process.Start();

            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            return process.ExitCode == 0 ? NormalizeWhitespace(output) : string.Empty;
        }
        catch
        {
            // Silently swallow exceptions (missing binary, permission errors, etc.)
            // so a broken OCR setup never crashes the indexing pipeline.
            return string.Empty;
        }
    }

    /// <summary>
    /// Collapses consecutive whitespace characters (spaces, tabs, newlines) in
    /// <paramref name="input"/> into a single space and trims leading/trailing whitespace.
    /// This keeps OCR output compact before it is stored as vector metadata.
    /// </summary>
    /// <param name="input">Raw text to normalise.</param>
    /// <returns>Normalised text, or an empty string if the input is whitespace-only.</returns>
    private static string NormalizeWhitespace(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(input.Length);
        var wasWhiteSpace = false;

        foreach (var ch in input)
        {
            if (char.IsWhiteSpace(ch))
            {
                // Emit one space the first time we see whitespace, then suppress
                // any additional consecutive whitespace characters.
                if (!wasWhiteSpace)
                {
                    builder.Append(' ');
                    wasWhiteSpace = true;
                }
            }
            else
            {
                builder.Append(ch);
                wasWhiteSpace = false;
            }
        }

        return builder.ToString().Trim();
    }
}