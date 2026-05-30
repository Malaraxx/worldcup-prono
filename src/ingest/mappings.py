"""
Mappings centralisés : noms d'équipes, slugs Transfermarkt, pages Wikipedia, etc.
"""

# ── Noms d'équipes martj42 → convention projet ────────────────────────────────
# Vide : martj42 et Wikipedia WC2026 utilisent déjà les mêmes noms communs.
TEAM_NAME_MAP: dict[str, str] = {}

# ── Pages Wikipedia équipes nationales ───────────────────────────────────────
# Pattern par défaut : "{team}_national_football_team"
# Exceptions ci-dessous (page name sans espaces = underscores déjà gérés dans le code).
WIKI_NATIONAL_TEAM_PAGE: dict[str, str] = {
    "United States": "United_States_men%27s_national_soccer_team",
}

# ── Code pays club → nom de championnat (pour colonne `league`) ──────────────
# Source : codes clubnat Transfermarkt/Wikipedia (ISO 3166-1 alpha-3 ou codes FIFA)
CLUBNAT_TO_LEAGUE: dict[str, str] = {
    # Big 5
    "ENG": "Premier League",
    "ESP": "La Liga",
    "GER": "Bundesliga",
    "ITA": "Serie A",
    "FRA": "Ligue 1",
    # Autres ligues majeures
    "POR": "Primeira Liga",
    "NED": "Eredivisie",
    "BEL": "Belgian First Division A",
    "SCO": "Scottish Premiership",
    "TUR": "Süper Lig",
    "RUS": "Russian Premier League",
    "UKR": "Ukrainian Premier League",
    "GRE": "Super League Greece",
    "AUT": "Austrian Football Bundesliga",
    "SUI": "Swiss Super League",
    "DEN": "Danish Superliga",
    "NOR": "Eliteserien",
    "SWE": "Allsvenskan",
    "CRO": "HNL",
    "SRB": "Serbian SuperLiga",
    # Amériques
    "BRA": "Campeonato Brasileiro Série A",
    "ARG": "Liga Profesional de Fútbol",
    "MEX": "Liga MX",
    "USA": "MLS",
    "COL": "Categoría Primera A",
    "CHI": "Primera División de Chile",
    "URU": "Primera División Uruguaya",
    "ECU": "Serie A Ecuador",
    "PAR": "División Profesional",
    # Asie / Moyen-Orient
    "SAU": "Saudi Pro League",
    "QAT": "Qatar Stars League",
    "UAE": "UAE Pro League",
    "JPN": "J1 League",
    "KOR": "K League 1",
    "CHN": "Chinese Super League",
    "AUS": "A-League",
    # Afrique
    "MAR": "Botola Pro",
    "EGY": "Egyptian Premier League",
    "TUN": "Ligue Professionnelle 1 Tunisienne",
    "ALG": "Ligue Professionnelle 1 Algérienne",
    "SEN": "Ligue 1 Sénégal",
    "GHA": "Ghana Premier League",
    "CIV": "Ligue 1 Côte d'Ivoire",
    "CMR": "MTN Elite One",
    "NGA": "Nigeria Premier Football League",
    # Autres
    "MKD": "First Football League of Macedonia",
    "SVK": "Slovak Super Liga",
    "SVN": "Slovenian PrvaLiga",
    "HUN": "OTP Bank Liga",
    "CZE": "Czech First League",
    "POL": "Ekstraklasa",
    "ROU": "Liga I",
}

# ── Mapping équipe → (slug, id) Transfermarkt ────────────────────────────────
# À valider lors de l'Étape 5 via web_fetch Transfermarkt
TRANSFERMARKT_TEAMS: dict[str, tuple[str, int]] = {}
