"""Descarga los CSV de football-data.co.uk a data/raw/.

Uso (desde la raíz del repo):
    python scripts/download_data.py                      # Premier League 2015/16 a 2025/26
    python scripts/download_data.py --start 2018 --end 2025
    python scripts/download_data.py --include-current    # agrega la temporada en curso (parcial); siempre se vuelve a bajar
    python scripts/download_data.py --fixtures           # baja también los próximos partidos (fixtures.csv)
    python scripts/download_data.py --league SP1         # otra liga (SP1 = LaLiga)

Solo usa la librería estándar de Python. No sobrescribe archivos ya descargados salvo con --force.
"""
import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{league}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CURRENT_SEASON_START = 2026  # temporada 2026/27, en curso (parcial)


def season_code(start_year: int) -> str:
    """2025 -> '2526' (formato que usa football-data.co.uk)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sports-predictor)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        print(f"  ERROR HTTP {e.code} en {url}")
        return False
    except urllib.error.URLError as e:
        print(f"  ERROR de red en {url}: {e.reason}")
        return False
    # Validación mínima: debe parecer un CSV de partidos.
    if b"HomeTeam" not in data[:2000]:
        print(f"  ERROR: {url} no parece un CSV de partidos (¿cambió el formato o la URL?)")
        return False
    dest.write_bytes(data)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--league", default="E0", help="Código de liga (E0 = Premier League)")
    p.add_argument("--start", type=int, default=2015, help="Año de inicio de la primera temporada (2015 = 2015/16)")
    p.add_argument("--end", type=int, default=2025, help="Año de inicio de la última temporada (2025 = 2025/26)")
    p.add_argument("--include-current", action="store_true", help="Incluye la temporada en curso (parcial)")
    p.add_argument("--fixtures", action="store_true", help="Baja también los próximos partidos (fixtures.csv)")
    p.add_argument("--force", action="store_true", help="Vuelve a descargar aunque el archivo exista")
    args = p.parse_args()

    end = CURRENT_SEASON_START if args.include_current else args.end
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for year in range(args.start, end + 1):
        code = season_code(year)
        dest = RAW_DIR / f"{args.league}_{code}.csv"
        # La temporada en curso cambia cada semana: siempre se vuelve a bajar.
        if dest.exists() and not args.force and year != CURRENT_SEASON_START:
            print(f"{dest.name}: ya existe, se omite")
            continue
        print(f"{dest.name}: descargando...")
        if download(BASE_URL.format(code=code, league=args.league), dest):
            print(f"  ok ({dest.stat().st_size / 1024:.0f} KB)")
        else:
            failed.append(code)
        time.sleep(0.5)  # cortesía con el servidor

    if args.fixtures:
        print("fixtures.csv: descargando...")
        if not download(FIXTURES_URL, RAW_DIR / "fixtures.csv"):
            failed.append("fixtures")

    if failed:
        print(f"\nFallaron las descargas: {', '.join(failed)}")
        print("Plan B: baja los CSV a mano desde https://www.football-data.co.uk/englandm.php")
        print("y guárdalos en data/raw/ con el nombre E0_<temporada>.csv (ejemplo: E0_2526.csv).")
        return 1
    print("\nListo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
