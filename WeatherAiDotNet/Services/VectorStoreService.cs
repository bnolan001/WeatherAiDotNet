using System.Text.Json;
using Microsoft.Data.Sqlite;
using WeatherAiDotNet.Models;

namespace WeatherAiDotNet.Services;

/// <summary>
/// Manages the local SQLite vector database that stores chunk embeddings and
/// document fingerprints.  Provides schema creation, incremental document indexing,
/// vector storage, and hybrid (semantic + lexical) retrieval.
/// </summary>
/// <remarks>
/// <para>
/// <b>Schema overview</b>
/// <list type="bullet">
///   <item><description>
///     <c>rag_vectors</c> – one row per chunk or image-item containing the text and
///     its JSON-serialised embedding vector.
///   </description></item>
///   <item><description>
///     <c>rag_documents</c> – one row per indexed PDF.  Stores a lightweight
///     fingerprint (file size + last-write timestamp) so unchanged PDFs can be
///     skipped on subsequent runs without re-reading or re-embedding them.
///   </description></item>
/// </list>
/// </para>
/// <para>
/// <b>Hybrid retrieval</b> combines cosine similarity (80 %) with a lexical overlap
/// score (20 %) so the ranking is robust even when the embedding model performs
/// poorly on domain-specific terminology.
/// </para>
/// </remarks>
public static class VectorStoreService
{
    /// <summary>
    /// Creates the SQLite database file and the required tables/indexes if they do
    /// not already exist.  Safe to call on every startup — all DDL statements use
    /// <c>CREATE … IF NOT EXISTS</c>.
    /// </summary>
    /// <param name="dbPath">Path to the SQLite database file.  Directories are created automatically.</param>
    public static void EnsureDatabase(string dbPath)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(dbPath));
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var command = connection.CreateCommand();
        command.CommandText =
            """
            CREATE TABLE IF NOT EXISTS rag_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector_json TEXT NOT NULL
            );

            -- Index on collection speeds up all queries that filter by dataset name.
            CREATE INDEX IF NOT EXISTS idx_rag_vectors_collection ON rag_vectors (collection);

            -- Index on (collection, source) speeds up per-document deletes during re-indexing.
            CREATE INDEX IF NOT EXISTS idx_rag_vectors_source ON rag_vectors (collection, source);

            CREATE TABLE IF NOT EXISTS rag_documents (
                collection TEXT NOT NULL,
                source TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                indexed_utc TEXT NOT NULL,
                PRIMARY KEY (collection, source)
            );
            """;

        command.ExecuteNonQuery();
    }

    /// <summary>
    /// Checks whether the stored fingerprint for a document matches the one computed
    /// from the file on disk.  A match means the file has not changed since it was
    /// last indexed, so it can be skipped.
    /// </summary>
    /// <param name="dbPath">Path to the SQLite database file.</param>
    /// <param name="collectionName">The collection that owns the document.</param>
    /// <param name="source">Relative file path used as the document identifier.</param>
    /// <param name="fingerprint">The fingerprint computed from the current file (size:ticks).</param>
    /// <returns>
    /// <see langword="true"/> if the stored fingerprint equals <paramref name="fingerprint"/>;
    /// <see langword="false"/> if the document is new, modified, or missing from the database.
    /// </returns>
    public static bool IsDocumentUpToDate(string dbPath, string collectionName, string source, string fingerprint)
    {
        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT fingerprint
            FROM rag_documents
            WHERE collection = $collection AND source = $source
            """;
        command.Parameters.AddWithValue("$collection", collectionName);
        command.Parameters.AddWithValue("$source", source);

        var existing = command.ExecuteScalar() as string;
        return string.Equals(existing, fingerprint, StringComparison.Ordinal);
    }

    /// <summary>
    /// Atomically replaces all stored vectors for a single document and updates its
    /// fingerprint record.  Runs inside a transaction so a failed write never leaves
    /// the database in a partially updated state.
    /// </summary>
    /// <param name="dbPath">Path to the SQLite database file.</param>
    /// <param name="collectionName">The collection that owns the document.</param>
    /// <param name="source">Relative file path of the document being replaced.</param>
    /// <param name="fingerprint">New fingerprint to record for this document.</param>
    /// <param name="vectors">
    /// All chunks (text + vector) produced from the current version of the document.
    /// </param>
    public static void ReplaceDocumentVectors(
        string dbPath,
        string collectionName,
        string source,
        string fingerprint,
        IReadOnlyList<StoredVector> vectors)
    {
        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var transaction = connection.BeginTransaction();

        // Step 1: delete all existing chunks for this document so stale data
        // from a previously longer version of the PDF is not left behind.
        using (var deleteVectors = connection.CreateCommand())
        {
            deleteVectors.Transaction = transaction;
            deleteVectors.CommandText =
                """
                DELETE FROM rag_vectors
                WHERE collection = $collection AND source = $source
                """;
            deleteVectors.Parameters.AddWithValue("$collection", collectionName);
            deleteVectors.Parameters.AddWithValue("$source", source);
            deleteVectors.ExecuteNonQuery();
        }

        // Step 2: insert the new chunks.  The vector is stored as a JSON array
        // because SQLite has no native vector type.
        foreach (var item in vectors)
        {
            using var insert = connection.CreateCommand();
            insert.Transaction = transaction;
            insert.CommandText =
                """
                INSERT INTO rag_vectors (collection, source, chunk_index, text, vector_json)
                VALUES ($collection, $source, $chunkIndex, $text, $vectorJson)
                """;

            insert.Parameters.AddWithValue("$collection", item.Collection);
            insert.Parameters.AddWithValue("$source", item.Source);
            insert.Parameters.AddWithValue("$chunkIndex", item.ChunkIndex);
            insert.Parameters.AddWithValue("$text", item.Text);
            insert.Parameters.AddWithValue("$vectorJson", JsonSerializer.Serialize(item.Vector));
            insert.ExecuteNonQuery();
        }

        // Step 3: upsert the document fingerprint so future runs can skip this file.
        using (var upsertDocument = connection.CreateCommand())
        {
            upsertDocument.Transaction = transaction;
            upsertDocument.CommandText =
                """
                INSERT INTO rag_documents (collection, source, fingerprint, indexed_utc)
                VALUES ($collection, $source, $fingerprint, $indexedUtc)
                ON CONFLICT(collection, source) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    indexed_utc = excluded.indexed_utc
                """;

            upsertDocument.Parameters.AddWithValue("$collection", collectionName);
            upsertDocument.Parameters.AddWithValue("$source", source);
            upsertDocument.Parameters.AddWithValue("$fingerprint", fingerprint);
            upsertDocument.Parameters.AddWithValue("$indexedUtc", DateTime.UtcNow.ToString("O"));
            upsertDocument.ExecuteNonQuery();
        }

        transaction.Commit();
    }

    /// <summary>
    /// Returns <see langword="true"/> when at least one vector exists in the given
    /// collection.  Used as a guard before starting the Q&amp;A loop to avoid
    /// querying an empty database.
    /// </summary>
    public static bool HasAnyVectors(string dbPath, string collectionName)
    {
        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var command = connection.CreateCommand();
        command.CommandText = "SELECT COUNT(1) FROM rag_vectors WHERE collection = $collection";
        command.Parameters.AddWithValue("$collection", collectionName);

        var count = Convert.ToInt32(command.ExecuteScalar());
        return count > 0;
    }

    /// <summary>
    /// Retrieves the most relevant chunks for a user question using hybrid retrieval:
    /// a cosine similarity score (semantic relevance) blended with a lexical overlap
    /// score (keyword matching).
    /// </summary>
    /// <remarks>
    /// The method loads <b>all</b> vectors for the collection into memory, scores
    /// each one, and returns the top-K results.  For a proof-of-concept with a
    /// moderate number of PDFs this is acceptable; a production system would use an
    /// ANN index (e.g., HNSW) to avoid the linear scan.
    /// </remarks>
    /// <param name="dbPath">Path to the SQLite database file.</param>
    /// <param name="collectionName">Collection to search.</param>
    /// <param name="question">Raw question text used for lexical scoring.</param>
    /// <param name="queryVector">Pre-computed embedding of the question.</param>
    /// <param name="topK">Number of chunks to return after final re-ranking.</param>
    /// <param name="retrievalPool">
    /// Candidate pool size: the top <paramref name="retrievalPool"/> results are
    /// selected first, then re-ranked down to <paramref name="topK"/>.  A larger
    /// pool improves recall at the cost of extra work.
    /// </param>
    public static List<SearchMatch> Search(string dbPath, string collectionName, string question, float[] queryVector, int topK, int retrievalPool)
    {
        using var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();

        using var command = connection.CreateCommand();

        // Load all vectors for the collection; scoring is done in C#.
        command.CommandText =
            """
            SELECT source, chunk_index, text, vector_json
            FROM rag_vectors
            WHERE collection = $collection
            """;
        command.Parameters.AddWithValue("$collection", collectionName);

        var scored = new List<ScoredCandidate>();

        // Pre-compute query features once because they are reused for every chunk.
        var queryTerms = Tokenize(question).ToHashSet(StringComparer.Ordinal);
        var normalizedQuestionPhrase = NormalizePhrase(question);

        // For short or number-heavy questions (common in weather references, tables,
        // and definitions), lexical matching tends to be more reliable than pure vector
        // similarity, so we rebalance weights toward lexical evidence.
        var isShortOrNumericQuery = queryTerms.Count <= 5 || queryTerms.Any(static term => term.Any(char.IsDigit));
        var semanticWeight = isShortOrNumericQuery ? 0.6f : 0.78f;
        var lexicalWeight = 1f - semanticWeight;

        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            var source = reader.GetString(0);
            var chunkIndex = reader.GetInt32(1);
            var text = reader.GetString(2);
            var vectorJson = reader.GetString(3);

            var vector = JsonSerializer.Deserialize<float[]>(vectorJson);

            // Skip rows whose vectors have a different dimension (e.g., left over
            // from a previous run with a different embedding model).
            if (vector is null || vector.Length != queryVector.Length)
            {
                continue;
            }

            // Cosine similarity: works correctly because vectors are L2-normalised at
            // ingestion time, so the dot product equals the cosine.
            var cosine = CosineSimilarity(queryVector, vector);

            // Lexical overlap favors chunks that contain the same key terms as the query.
            var lexicalOverlap = LexicalOverlapScore(queryTerms, text);

            // Frequency adds extra signal when a critical term appears multiple times in
            // a chunk (for example acronyms, codes, or repeated table headers).
            var lexicalFrequency = LexicalFrequencyScore(queryTerms, text);

            // Explicit phrase matches are a strong indicator for factoid questions, so we
            // apply a small additive boost instead of overwhelming the semantic score.
            var phraseBoost = HasPhraseMatch(text, normalizedQuestionPhrase) ? 0.08f : 0f;

            var lexical = (0.7f * lexicalOverlap) + (0.3f * lexicalFrequency);
            var score = (semanticWeight * cosine) + (lexicalWeight * lexical) + phraseBoost;

            scored.Add(new ScoredCandidate(source, chunkIndex, text, score, vector));
        }

        if (scored.Count == 0)
        {
            return [];
        }

        // Stage 1 (recall): pull a wider candidate set to reduce the chance of missing
        // answer-bearing chunks that are not in the first few ranks.
        var safeTopK = Math.Max(1, topK);
        var candidatePoolSize = Math.Max(Math.Max(retrievalPool, safeTopK * 4), 20);
        var candidatePool = scored
            .OrderByDescending(item => item.Score)
            .Take(candidatePoolSize)
            .ToList();

        // Stage 2 (precision/diversity): rerank with a small redundancy penalty so the
        // final prompt context includes different parts of the corpus instead of near-
        // duplicate chunks that repeat the same sentence.
        var diversityPenalty = isShortOrNumericQuery ? 0.1f : 0.18f;
        return RerankWithDiversity(candidatePool, safeTopK, diversityPenalty)
            .Select(static item => new SearchMatch(item.Source, item.ChunkIndex, item.Text, item.Score))
            .ToList();
    }

    /// <summary>
    /// Computes a lexical overlap score between the query terms and the chunk text.
    /// Returns the fraction of query tokens that are present in the chunk (Jaccard-like).
    /// </summary>
    private static float LexicalOverlapScore(HashSet<string> queryTerms, string text)
    {
        if (queryTerms.Count == 0)
        {
            return 0f;
        }

        var chunkTerms = Tokenize(text).ToHashSet(StringComparer.Ordinal);
        if (chunkTerms.Count == 0)
        {
            return 0f;
        }

        var intersection = queryTerms.Count(t => chunkTerms.Contains(t));
        return intersection / (float)queryTerms.Count;
    }

    /// <summary>
    /// Computes a frequency-aware lexical score where repeated query terms in a chunk
    /// modestly increase confidence.  This improves retrieval for fact lookups where
    /// key terms may appear multiple times in structured text.
    /// </summary>
    private static float LexicalFrequencyScore(HashSet<string> queryTerms, string text)
    {
        if (queryTerms.Count == 0)
        {
            return 0f;
        }

        var occurrences = 0;
        foreach (var term in Tokenize(text))
        {
            if (queryTerms.Contains(term))
            {
                occurrences++;
            }
        }

        // Saturate the score to [0, 1] so frequent repeats help ranking but do not
        // dominate semantic similarity entirely.
        var normalizer = Math.Max(1, queryTerms.Count * 2);
        return Math.Min(1f, occurrences / (float)normalizer);
    }

    /// <summary>
    /// Performs a second-stage rerank using Max Marginal Relevance (MMR).
    /// Keeps highly relevant chunks while discouraging near-duplicates in the final set.
    /// </summary>
    private static List<ScoredCandidate> RerankWithDiversity(
        IReadOnlyList<ScoredCandidate> candidates,
        int topK,
        float diversityPenalty)
    {
        var selected = new List<ScoredCandidate>(topK);
        var remaining = candidates.ToList();

        while (selected.Count < topK && remaining.Count > 0)
        {
            ScoredCandidate? best = null;
            var bestScore = float.NegativeInfinity;

            foreach (var candidate in remaining)
            {
                var maxSimilarityToSelected = 0f;
                foreach (var existing in selected)
                {
                    maxSimilarityToSelected = Math.Max(maxSimilarityToSelected, CosineSimilarity(candidate.Vector, existing.Vector));
                }

                var mmrScore = candidate.Score - (diversityPenalty * maxSimilarityToSelected);
                if (mmrScore > bestScore)
                {
                    best = candidate;
                    bestScore = mmrScore;
                }
            }

            if (best is null)
            {
                break;
            }

            selected.Add(best);
            _ = remaining.Remove(best);
        }

        return selected;
    }

    /// <summary>
    /// Normalizes text into a token phrase used by phrase containment checks.
    /// Token-based normalization is resilient to punctuation and whitespace differences.
    /// </summary>
    private static string NormalizePhrase(string input)
        => string.Join(' ', Tokenize(input));

    /// <summary>
    /// Returns <see langword="true"/> when the normalized query phrase appears in the
    /// normalized chunk text.
    /// </summary>
    private static bool HasPhraseMatch(string text, string normalizedQuestionPhrase)
    {
        if (string.IsNullOrWhiteSpace(normalizedQuestionPhrase))
        {
            return false;
        }

        var normalizedText = NormalizePhrase(text);
        return normalizedText.Contains(normalizedQuestionPhrase, StringComparison.Ordinal);
    }

    /// <summary>
    /// Computes the dot product of two equal-length vectors.  Because stored vectors
    /// are L2-normalised before insertion, the dot product equals the cosine similarity
    /// and ranges from -1 (opposite) to 1 (identical direction).
    /// </summary>
    private static float CosineSimilarity(float[] left, float[] right)
    {
        if (left.Length != right.Length)
        {
            return 0f;
        }

        float dot = 0;
        for (var i = 0; i < left.Length; i++)
        {
            dot += left[i] * right[i];
        }

        return dot;
    }

    /// <summary>
    /// Tokenises text into lowercase alphanumeric tokens of length ≥ 2.
    /// Used for both query expansion and chunk keyword extraction.
    /// </summary>
    private static IEnumerable<string> Tokenize(string input)
    {
        var sb = new System.Text.StringBuilder();

        foreach (var ch in input)
        {
            if (char.IsLetterOrDigit(ch))
            {
                sb.Append(char.ToLowerInvariant(ch));
                continue;
            }

            if (sb.Length > 0)
            {
                var token = sb.ToString();
                if (token.Length >= 2)
                {
                    yield return token;
                }

                sb.Clear();
            }
        }

        if (sb.Length > 0)
        {
            var token = sb.ToString();
            if (token.Length >= 2)
            {
                yield return token;
            }
        }
    }

    private sealed record ScoredCandidate(string Source, int ChunkIndex, string Text, float Score, float[] Vector);
}