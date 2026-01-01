# src/rag_semantic/rank_compiler.py
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


# -----------------------------
# Parsing helpers
# -----------------------------
def _parse_ordinal_int(token: str) -> Optional[int]:
    if token is None:
        return None
    token = token.strip().lower()
    m = re.match(r"^(\d+)(?:st|nd|rd|th)?$", token)
    if not m:
        return None
    n = int(m.group(1))
    return n if n >= 1 else None


def _strip_month_modifier(question: str) -> Tuple[str, bool]:
    """
    Detects and strips month modifiers like:
      - "by month", "per month", "monthly", "each month"
    Returns (cleaned_question, by_month_flag)
    """
    q = (question or "").strip()
    q_norm = re.sub(r"\s+", " ", q).strip()
    q_low = q_norm.lower()

    by_month = False

    # common month modifier phrases
    month_patterns = [
        r"\bby month\b",
        r"\bper month\b",
        r"\beach month\b",
        r"\bmonthly\b",
        r"\bmonth-wise\b",
        r"\bmonth wise\b",
    ]

    for pat in month_patterns:
        if re.search(pat, q_low):
            by_month = True
            q_low = re.sub(pat, " ", q_low)
            q_low = re.sub(r"\s+", " ", q_low).strip()

    # return cleaned original casing is not necessary since we parse lower anyway
    return q_low, by_month


def parse_nth_inner_in_mth_outer(question: str) -> Optional[Dict[str, object]]:
    """
    Parses questions like:
      - "10th genre in 5th country"
      - "10 genre in 5 country"
      - "10th track for 5th artist"
      - "nth track for mth artist"  (only numeric ordinals supported)
    Also supports month modifier, e.g.:
      - "10th track for 5th artist by month"
    Returns:
      {"inner": "...", "outer": "...", "n": int, "m": int, "preposition": "in"/"for", "by_month": bool}
    """
    q_low, by_month = _strip_month_modifier(question)

    # normalize multiple spaces
    q = re.sub(r"\s+", " ", q_low)

    # Accept: "<n> <inner> in/for <m> <outer>"
    # inner/outer are single words here (genre, country, track, artist, album)
    m = re.search(
        r"\b(\d+(?:st|nd|rd|th)?)\s+([a-z_]+)\s+(in|for)\s+(?:the\s+)?(\d+(?:st|nd|rd|th)?)\s+([a-z_]+)\b",
        q,
    )
    if not m:
        return None

    n = _parse_ordinal_int(m.group(1))
    inner = m.group(2)
    prep = m.group(3)
    mth = _parse_ordinal_int(m.group(4))
    outer = m.group(5)

    if not n or not mth:
        return None

    # Basic plural normalization
    if inner.endswith("s"):
        inner = inner[:-1]
    if outer.endswith("s"):
        outer = outer[:-1]

    return {"inner": inner, "outer": outer, "n": n, "m": mth, "preposition": prep, "by_month": by_month}


# -----------------------------
# Common month-safe invoice CTE
# -----------------------------
def _invoice_month_cte(alias: str = "i") -> str:
    """
    Produces a reusable safe timestamp + month expression.
    We filter to sane range to avoid DuckDB overflow/BC dates.
    """
    return (
        "WITH clean_invoice AS (\n"
        f"  SELECT {alias}.InvoiceId,\n"
        f"         try_cast({alias}.InvoiceDate AS TIMESTAMP) AS InvoiceTS\n"
        f"  FROM source.Invoice {alias}\n"
        "  WHERE try_cast(InvoiceDate AS TIMESTAMP) IS NOT NULL\n"
        "    AND try_cast(InvoiceDate AS TIMESTAMP) >= TIMESTAMP '1900-01-01'\n"
        "    AND try_cast(InvoiceDate AS TIMESTAMP) <  TIMESTAMP '2100-01-01'\n"
        ")\n"
    )


# -----------------------------
# Compilers
# -----------------------------
def _compile_genre_in_country(n: int, mth_country: int, by_month: bool) -> str:
    # Nth genre in Mth country, country ranked by total revenue.
    # If by_month=True: compute monthly ranks within the selected country.

    if not by_month:
        return (
            "WITH country_totals AS (\n"
            "  SELECT i.BillingCountry AS Country,\n"
            "         SUM(il.UnitPrice * il.Quantity) AS CountryRevenue\n"
            "  FROM source.Invoice i\n"
            "  JOIN source.InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
            "  GROUP BY i.BillingCountry\n"
            "),\n"
            "ranked_countries AS (\n"
            "  SELECT Country,\n"
            "         CountryRevenue,\n"
            "         ROW_NUMBER() OVER (ORDER BY CountryRevenue DESC, Country ASC) AS CountryRank\n"
            "  FROM country_totals\n"
            "),\n"
            "target_country AS (\n"
            f"  SELECT Country, CountryRevenue\n"
            f"  FROM ranked_countries\n"
            f"  WHERE CountryRank = {mth_country}\n"
            "),\n"
            "country_genre AS (\n"
            "  SELECT tc.Country,\n"
            "         tc.CountryRevenue,\n"
            "         g.Name AS Genre,\n"
            "         SUM(il.UnitPrice * il.Quantity) AS GenreRevenue\n"
            "  FROM target_country tc\n"
            "  JOIN source.Invoice i ON i.BillingCountry = tc.Country\n"
            "  JOIN source.InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
            "  JOIN source.Track t ON il.TrackId = t.TrackId\n"
            "  JOIN source.Genre g ON t.GenreId = g.GenreId\n"
            "  GROUP BY tc.Country, tc.CountryRevenue, g.Name\n"
            "),\n"
            "ranked_genres AS (\n"
            "  SELECT Country,\n"
            "         CountryRevenue,\n"
            "         Genre,\n"
            "         GenreRevenue,\n"
            "         ROW_NUMBER() OVER (PARTITION BY Country ORDER BY GenreRevenue DESC, Genre ASC) AS GenreRank\n"
            "  FROM country_genre\n"
            ")\n"
            "SELECT Country,\n"
            "       Genre,\n"
            "       GenreRevenue AS Revenue\n"
            "FROM ranked_genres\n"
            f"WHERE GenreRank = {n}\n"
            "ORDER BY CountryRevenue DESC, Country ASC, Revenue DESC, Genre ASC"
        )

    # by month
    return (
        "WITH country_totals AS (\n"
        "  SELECT i.BillingCountry AS Country,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS CountryRevenue\n"
        "  FROM source.Invoice i\n"
        "  JOIN source.InvoiceLine il ON i.InvoiceId = il.InvoiceId\n"
        "  GROUP BY i.BillingCountry\n"
        "),\n"
        "ranked_countries AS (\n"
        "  SELECT Country,\n"
        "         CountryRevenue,\n"
        "         ROW_NUMBER() OVER (ORDER BY CountryRevenue DESC, Country ASC) AS CountryRank\n"
        "  FROM country_totals\n"
        "),\n"
        "target_country AS (\n"
        f"  SELECT Country, CountryRevenue\n"
        f"  FROM ranked_countries\n"
        f"  WHERE CountryRank = {mth_country}\n"
        "),\n"
        "clean_invoice AS (\n"
        "  SELECT i.InvoiceId,\n"
        "         i.BillingCountry AS Country,\n"
        "         try_cast(i.InvoiceDate AS TIMESTAMP) AS InvoiceTS\n"
        "  FROM source.Invoice i\n"
        "  WHERE try_cast(i.InvoiceDate AS TIMESTAMP) IS NOT NULL\n"
        "    AND try_cast(i.InvoiceDate AS TIMESTAMP) >= TIMESTAMP '1900-01-01'\n"
        "    AND try_cast(i.InvoiceDate AS TIMESTAMP) <  TIMESTAMP '2100-01-01'\n"
        "),\n"
        "country_genre_monthly AS (\n"
        "  SELECT tc.Country,\n"
        "         tc.CountryRevenue,\n"
        "         strftime(ci.InvoiceTS, '%Y-%m') AS Month,\n"
        "         g.Name AS Genre,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS GenreRevenue\n"
        "  FROM target_country tc\n"
        "  JOIN clean_invoice ci ON ci.Country = tc.Country\n"
        "  JOIN source.InvoiceLine il ON il.InvoiceId = ci.InvoiceId\n"
        "  JOIN source.Track t ON t.TrackId = il.TrackId\n"
        "  JOIN source.Genre g ON g.GenreId = t.GenreId\n"
        "  GROUP BY tc.Country, tc.CountryRevenue, Month, g.Name\n"
        "),\n"
        "ranked AS (\n"
        "  SELECT Country,\n"
        "         CountryRevenue,\n"
        "         Month,\n"
        "         Genre,\n"
        "         GenreRevenue,\n"
        "         ROW_NUMBER() OVER (PARTITION BY Month ORDER BY GenreRevenue DESC, Genre ASC) AS GenreRank\n"
        "  FROM country_genre_monthly\n"
        ")\n"
        "SELECT Country,\n"
        "       Month,\n"
        "       Genre,\n"
        "       GenreRevenue AS Revenue\n"
        "FROM ranked\n"
        f"WHERE GenreRank = {n}\n"
        "ORDER BY Month ASC, Revenue DESC, Genre ASC"
    )


def _compile_track_for_artist(n: int, mth_artist: int, by_month: bool) -> str:
    # Nth track for Mth artist (artist ranked by revenue).
    # If by_month=True: compute monthly ranks within that artist.

    if not by_month:
        return (
            "WITH artist_totals AS (\n"
            "  SELECT a.ArtistId,\n"
            "         a.Name AS Artist,\n"
            "         SUM(il.UnitPrice * il.Quantity) AS ArtistRevenue\n"
            "  FROM source.Artist a\n"
            "  JOIN source.Album al ON al.ArtistId = a.ArtistId\n"
            "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
            "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
            "  GROUP BY a.ArtistId, a.Name\n"
            "),\n"
            "ranked_artists AS (\n"
            "  SELECT ArtistId,\n"
            "         Artist,\n"
            "         ArtistRevenue,\n"
            "         ROW_NUMBER() OVER (ORDER BY ArtistRevenue DESC, Artist ASC) AS ArtistRank\n"
            "  FROM artist_totals\n"
            "),\n"
            "target_artist AS (\n"
            f"  SELECT ArtistId, Artist, ArtistRevenue\n"
            f"  FROM ranked_artists\n"
            f"  WHERE ArtistRank = {mth_artist}\n"
            "),\n"
            "artist_track AS (\n"
            "  SELECT ta.Artist,\n"
            "         ta.ArtistRevenue,\n"
            "         t.TrackId,\n"
            "         t.Name AS Track,\n"
            "         SUM(il.UnitPrice * il.Quantity) AS TrackRevenue\n"
            "  FROM target_artist ta\n"
            "  JOIN source.Album al ON al.ArtistId = ta.ArtistId\n"
            "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
            "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
            "  GROUP BY ta.Artist, ta.ArtistRevenue, t.TrackId, t.Name\n"
            "),\n"
            "ranked_tracks AS (\n"
            "  SELECT Artist,\n"
            "         ArtistRevenue,\n"
            "         TrackId,\n"
            "         Track,\n"
            "         TrackRevenue,\n"
            "         ROW_NUMBER() OVER (PARTITION BY Artist ORDER BY TrackRevenue DESC, Track ASC) AS TrackRank\n"
            "  FROM artist_track\n"
            ")\n"
            "SELECT Artist,\n"
            "       Track,\n"
            "       TrackRevenue AS Revenue\n"
            "FROM ranked_tracks\n"
            f"WHERE TrackRank = {n}\n"
            "ORDER BY ArtistRevenue DESC, Artist ASC, Revenue DESC, Track ASC"
        )

    # by month
    return (
        "WITH artist_totals AS (\n"
        "  SELECT a.ArtistId,\n"
        "         a.Name AS Artist,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS ArtistRevenue\n"
        "  FROM source.Artist a\n"
        "  JOIN source.Album al ON al.ArtistId = a.ArtistId\n"
        "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
        "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
        "  GROUP BY a.ArtistId, a.Name\n"
        "),\n"
        "ranked_artists AS (\n"
        "  SELECT ArtistId,\n"
        "         Artist,\n"
        "         ArtistRevenue,\n"
        "         ROW_NUMBER() OVER (ORDER BY ArtistRevenue DESC, Artist ASC) AS ArtistRank\n"
        "  FROM artist_totals\n"
        "),\n"
        "target_artist AS (\n"
        f"  SELECT ArtistId, Artist, ArtistRevenue\n"
        f"  FROM ranked_artists\n"
        f"  WHERE ArtistRank = {mth_artist}\n"
        "),\n"
        "clean_invoice AS (\n"
        "  SELECT i.InvoiceId,\n"
        "         try_cast(i.InvoiceDate AS TIMESTAMP) AS InvoiceTS\n"
        "  FROM source.Invoice i\n"
        "  WHERE try_cast(i.InvoiceDate AS TIMESTAMP) IS NOT NULL\n"
        "    AND try_cast(i.InvoiceDate AS TIMESTAMP) >= TIMESTAMP '1900-01-01'\n"
        "    AND try_cast(i.InvoiceDate AS TIMESTAMP) <  TIMESTAMP '2100-01-01'\n"
        "),\n"
        "artist_track_monthly AS (\n"
        "  SELECT ta.Artist,\n"
        "         ta.ArtistRevenue,\n"
        "         strftime(ci.InvoiceTS, '%Y-%m') AS Month,\n"
        "         t.TrackId,\n"
        "         t.Name AS Track,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS TrackRevenue\n"
        "  FROM target_artist ta\n"
        "  JOIN source.Album al ON al.ArtistId = ta.ArtistId\n"
        "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
        "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
        "  JOIN clean_invoice ci ON ci.InvoiceId = il.InvoiceId\n"
        "  GROUP BY ta.Artist, ta.ArtistRevenue, Month, t.TrackId, t.Name\n"
        "),\n"
        "ranked AS (\n"
        "  SELECT Artist,\n"
        "         ArtistRevenue,\n"
        "         Month,\n"
        "         Track,\n"
        "         TrackRevenue,\n"
        "         ROW_NUMBER() OVER (PARTITION BY Month ORDER BY TrackRevenue DESC, Track ASC) AS TrackRank\n"
        "  FROM artist_track_monthly\n"
        ")\n"
        "SELECT Artist,\n"
        "       Month,\n"
        "       Track,\n"
        "       TrackRevenue AS Revenue\n"
        "FROM ranked\n"
        f"WHERE TrackRank = {n}\n"
        "ORDER BY Month ASC, Revenue DESC, Track ASC"
    )


def _compile_album_for_artist(n: int, mth_artist: int, by_month: bool) -> str:
    # Nth album for Mth artist. If by_month=True: album revenue per month.

    if not by_month:
        return (
            "WITH artist_totals AS (\n"
            "  SELECT a.ArtistId,\n"
            "         a.Name AS Artist,\n"
            "         SUM(il.UnitPrice * il.Quantity) AS ArtistRevenue\n"
            "  FROM source.Artist a\n"
            "  JOIN source.Album al ON al.ArtistId = a.ArtistId\n"
            "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
            "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
            "  GROUP BY a.ArtistId, a.Name\n"
            "),\n"
            "ranked_artists AS (\n"
            "  SELECT ArtistId,\n"
            "         Artist,\n"
            "         ArtistRevenue,\n"
            "         ROW_NUMBER() OVER (ORDER BY ArtistRevenue DESC, Artist ASC) AS ArtistRank\n"
            "  FROM artist_totals\n"
            "),\n"
            "target_artist AS (\n"
            f"  SELECT ArtistId, Artist, ArtistRevenue\n"
            f"  FROM ranked_artists\n"
            f"  WHERE ArtistRank = {mth_artist}\n"
            "),\n"
            "artist_album AS (\n"
            "  SELECT ta.Artist,\n"
            "         ta.ArtistRevenue,\n"
            "         al.AlbumId,\n"
            "         al.Title AS Album,\n"
            "         SUM(il.UnitPrice * il.Quantity) AS AlbumRevenue\n"
            "  FROM target_artist ta\n"
            "  JOIN source.Album al ON al.ArtistId = ta.ArtistId\n"
            "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
            "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
            "  GROUP BY ta.Artist, ta.ArtistRevenue, al.AlbumId, al.Title\n"
            "),\n"
            "ranked_albums AS (\n"
            "  SELECT Artist,\n"
            "         ArtistRevenue,\n"
            "         AlbumId,\n"
            "         Album,\n"
            "         AlbumRevenue,\n"
            "         ROW_NUMBER() OVER (PARTITION BY Artist ORDER BY AlbumRevenue DESC, Album ASC) AS AlbumRank\n"
            "  FROM artist_album\n"
            ")\n"
            "SELECT Artist,\n"
            "       Album,\n"
            "       AlbumRevenue AS Revenue\n"
            "FROM ranked_albums\n"
            f"WHERE AlbumRank = {n}\n"
            "ORDER BY ArtistRevenue DESC, Artist ASC, Revenue DESC, Album ASC"
        )

    # by month
    return (
        "WITH artist_totals AS (\n"
        "  SELECT a.ArtistId,\n"
        "         a.Name AS Artist,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS ArtistRevenue\n"
        "  FROM source.Artist a\n"
        "  JOIN source.Album al ON al.ArtistId = a.ArtistId\n"
        "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
        "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
        "  GROUP BY a.ArtistId, a.Name\n"
        "),\n"
        "ranked_artists AS (\n"
        "  SELECT ArtistId,\n"
        "         Artist,\n"
        "         ArtistRevenue,\n"
        "         ROW_NUMBER() OVER (ORDER BY ArtistRevenue DESC, Artist ASC) AS ArtistRank\n"
        "  FROM artist_totals\n"
        "),\n"
        "target_artist AS (\n"
        f"  SELECT ArtistId, Artist, ArtistRevenue\n"
        f"  FROM ranked_artists\n"
        f"  WHERE ArtistRank = {mth_artist}\n"
        "),\n"
        "clean_invoice AS (\n"
        "  SELECT i.InvoiceId,\n"
        "         try_cast(i.InvoiceDate AS TIMESTAMP) AS InvoiceTS\n"
        "  FROM source.Invoice i\n"
        "  WHERE try_cast(i.InvoiceDate AS TIMESTAMP) IS NOT NULL\n"
        "    AND try_cast(i.InvoiceDate AS TIMESTAMP) >= TIMESTAMP '1900-01-01'\n"
        "    AND try_cast(i.InvoiceDate AS TIMESTAMP) <  TIMESTAMP '2100-01-01'\n"
        "),\n"
        "artist_album_monthly AS (\n"
        "  SELECT ta.Artist,\n"
        "         ta.ArtistRevenue,\n"
        "         strftime(ci.InvoiceTS, '%Y-%m') AS Month,\n"
        "         al.AlbumId,\n"
        "         al.Title AS Album,\n"
        "         SUM(il.UnitPrice * il.Quantity) AS AlbumRevenue\n"
        "  FROM target_artist ta\n"
        "  JOIN source.Album al ON al.ArtistId = ta.ArtistId\n"
        "  JOIN source.Track t ON t.AlbumId = al.AlbumId\n"
        "  JOIN source.InvoiceLine il ON il.TrackId = t.TrackId\n"
        "  JOIN clean_invoice ci ON ci.InvoiceId = il.InvoiceId\n"
        "  GROUP BY ta.Artist, ta.ArtistRevenue, Month, al.AlbumId, al.Title\n"
        "),\n"
        "ranked AS (\n"
        "  SELECT Artist,\n"
        "         ArtistRevenue,\n"
        "         Month,\n"
        "         Album,\n"
        "         AlbumRevenue,\n"
        "         ROW_NUMBER() OVER (PARTITION BY Month ORDER BY AlbumRevenue DESC, Album ASC) AS AlbumRank\n"
        "  FROM artist_album_monthly\n"
        ")\n"
        "SELECT Artist,\n"
        "       Month,\n"
        "       Album,\n"
        "       AlbumRevenue AS Revenue\n"
        "FROM ranked\n"
        f"WHERE AlbumRank = {n}\n"
        "ORDER BY Month ASC, Revenue DESC, Album ASC"
    )


# -----------------------------
# Public entry point
# -----------------------------
def compile_rank_query(question: str) -> Optional[Tuple[str, str]]:
    """
    Returns (sql, explanation) if this question matches our Nth/Mth rank pattern and is supported.
    Now supports a generic month modifier: 'by month' / 'monthly' etc.
    """
    parsed = parse_nth_inner_in_mth_outer(question)
    if not parsed:
        return None

    inner = str(parsed["inner"])
    outer = str(parsed["outer"])
    n = int(parsed["n"])
    mth = int(parsed["m"])
    by_month = bool(parsed.get("by_month", False))

    if inner == "genre" and outer == "country":
        return (
            _compile_genre_in_country(n=n, mth_country=mth, by_month=by_month),
            (
                "Deterministic rank compiler: Nth genre within Mth country (by revenue)."
                + (" Month modifier applied (ranked per month)." if by_month else "")
            ),
        )

    if inner == "track" and outer == "artist":
        return (
            _compile_track_for_artist(n=n, mth_artist=mth, by_month=by_month),
            (
                "Deterministic rank compiler: Nth track for Mth artist (by revenue)."
                + (" Month modifier applied (ranked per month)." if by_month else "")
            ),
        )

    if inner == "album" and outer == "artist":
        return (
            _compile_album_for_artist(n=n, mth_artist=mth, by_month=by_month),
            (
                "Deterministic rank compiler: Nth album for Mth artist (by revenue)."
                + (" Month modifier applied (ranked per month)." if by_month else "")
            ),
        )

    return None
