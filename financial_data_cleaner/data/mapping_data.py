"""
Mapping data for financial field standardization.
"""

# Define standard field types
standard_field_types = {
    "ISIN": "string",
    "Fund Name": "string",
    "Share Name": "string",
    "NAV Date": "date",
    "NAV Value": "numeric",
    "Number of Shares": "numeric",
    "Currency": "string",
    "Compartment Assets": "numeric",
    "Compartment Currency": "string",
    "Coupon": "numeric",
    "Coupon Date": "date",
    "CIC Code": "string",
    "Previous NAV": "numeric",
    "Difference": "numeric",
    "Difference Percent": "numeric",
    "WPK Code": "string"
}

# Standard fields and their aliases
standard_fields = {
    "ISIN": ["code isin", "isin", "isin code", "code", "code_isin"],
    "Fund Name": ["fcp", "fund", "fund name", "nom du fonds", "libellé", "libelle", "fund_name", "name", "nom", "fund name"],
    "Share Name": ["share", "share name", "share_name", "part", "nom de la part"],
    "NAV Date": ["nav date", "date vl", "date", "date de publication", "nav_date", "date valeur", "valuation date", "navdate", "tradedate"],
    "NAV Value": ["valeur liquidative", "vl", "nav value", "nav", "value", "prix", "price", "valeur", "nav_today"],
    "Number of Shares": ["nb parts", "nombre de parts", "shares", "parts", "number of shares", "nb_parts", "shares_outstanding"],
    "Currency": ["devise", "currency", "ccy", "monnaie"],
    "Compartment Assets": ["encours net", "actif net", "compartment assets", "assets", "encours", "encours_global", "aum", "total_nav"],
    "Compartment Currency": ["compartment currency", "devise du compartiment"],
    "Coupon": ["coupon", "coupon rate"],
    "Coupon Date": ["coupon date", "date de coupon", "payment date"],
    "CIC Code": ["cic code", "cic"],
    "Previous NAV": ["nav previous", "previous nav", "nav previous day", "previous value"],
    "Difference": ["difference", "diff", "change"],
    "Difference Percent": ["difference %", "diff %", "change %", "percent change", "difference percent"],
    "WPK Code": ["wpk", "code_wpk", "wpk code"]
}

# Currency standardization mapping
currency_mapping = {
    '€': 'EUR', '$': 'USD', '£': 'GBP', 'EURO': 'EUR', 'EUROS': 'EUR',
    'US DOLLAR': 'USD', 'DOLLAR': 'USD', 'DOLLARS': 'USD', 'POUND': 'GBP',
    'YEN': 'JPY', 'JAPANESE YEN': 'JPY', 'SWISS FRANC': 'CHF', 'CHF': 'CHF',
    'CANADIAN DOLLAR': 'CAD', 'CAD': 'CAD', 'AUSTRALIAN DOLLAR': 'AUD', 'AUD': 'AUD'
}
