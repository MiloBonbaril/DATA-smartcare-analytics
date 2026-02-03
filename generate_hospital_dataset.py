"""
Synthetic hospital dataset generator (flows + HR + equipment), designed for scenario testing.

Important Windows note:
- Some environments enforce "application control" policies that block native extensions (.pyd/.dll).
  To ensure the script runs everywhere, this version uses **only the Python standard library**.
- CSV export is always supported.
- Parquet export is attempted only if `pyarrow` is importable (optional).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Literal, Optional

import random


ServiceName = Literal["Urgences", "Cardio", "Neuro", "Infectieuses", "Geriatrie", "Pediatrie"]


FR_DAYS = ("Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche")


@dataclass(frozen=True)
class ScenarioWindow:
    """Defines an inclusive [start, end] window in local time."""

    start: datetime
    end: datetime
    name: str

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts <= self.end


@dataclass(frozen=True)
class GeneratorConfig:
    """All tunables for the generator (explicit & reproducible)."""

    start: datetime
    end: datetime
    step: timedelta
    seed: int

    # Hospital "order of magnitude" constraints
    bed_capacity: int = 1800
    ed_passages_per_year: int = 100_000

    # Seasonality: amplitude expressed as % of base lambda
    seasonal_amplitude: float = 0.20
    seasonal_phase_days: int = -20  # shifts the sinus so winter is higher

    # EPI stock system (aggregate units)
    # Tuned to have more frequent, smaller deliveries and a less massive buffer.
    epi_initial_stock: int = 80_000
    epi_daily_delivery: int = 6_000  # baseline deliveries (split across the day)
    epi_emergency_threshold: int = 40_000
    epi_emergency_delivery: int = 30_000
    epi_emergency_lead_hours: int = 12

    # Scenarios
    epidemic_multiplier_infectious: float = 1.5
    heatwave_multiplier_geriatrics: float = 1.4
    strike_staff_multiplier: float = 0.7  # 30% reduction

    # Scenario windows (defaults are realistic placeholders)
    epidemic_windows: tuple[ScenarioWindow, ...] = ()
    heatwave_windows: tuple[ScenarioWindow, ...] = ()
    strike_windows: tuple[ScenarioWindow, ...] = ()


def _parse_datetime(s: str) -> datetime:
    """
    Accepts 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (quotes recommended in PowerShell).
    """
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m-%d":
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Invalid datetime format: {s!r}. Expected YYYY-MM-DD or YYYY-MM-DD HH:MM:SS.")


def _date_range_inclusive(start: datetime, end: datetime, step: timedelta) -> Iterable[datetime]:
    if step.total_seconds() <= 0:
        raise ValueError("Step must be positive.")
    cur = start
    while cur <= end:
        yield cur
        cur += step


def _seasonal_multiplier(day_of_year: int, amplitude: float, phase_days: int) -> float:
    """
    Sinusoidal seasonality on a 365-day period.
    Multiplier in [1-amplitude, 1+amplitude] (bounded below for safety).
    """
    angle = (2.0 * math.pi * (day_of_year + phase_days)) / 365.0
    mult = 1.0 + amplitude * math.sin(angle)
    return max(mult, 0.05)


def _hourly_pattern(hour: int, peak_hour: int, sharpness: float = 2.2) -> float:
    """
    Smooth intra-day pattern ~ [0.7, 1.4], centered on `peak_hour` (wrapped on 24h).
    """
    dist = min((hour - peak_hour) % 24, (peak_hour - hour) % 24)
    x = (dist / 12.0) * math.pi
    curve = (math.cos(x) + 1.0) / 2.0  # 1 at peak, 0 at opposite
    return 0.7 + 0.7 * (curve**sharpness)


def _dow_multiplier(dow_idx: int, service: ServiceName) -> float:
    """
    Day-of-week effect:
    - Urgences: slightly higher on weekends.
    - Scheduled services: lower on weekends.
    """
    is_weekend = 1 if dow_idx >= 5 else 0
    if service == "Urgences":
        return 1.0 + 0.10 * is_weekend
    return 1.0 - 0.25 * is_weekend


def _default_windows_for_years(years: Iterable[int]) -> tuple[
    tuple[ScenarioWindow, ...],
    tuple[ScenarioWindow, ...],
    tuple[ScenarioWindow, ...],
]:
    epidemics: list[ScenarioWindow] = []
    heatwaves: list[ScenarioWindow] = []
    strikes: list[ScenarioWindow] = []
    for y in years:
        # Epidemics & heatwaves stay mostly structured (predictable seasonal patterns).
        epidemics.append(
            ScenarioWindow(
                start=datetime(y, 2, 1, 0, 0, 0),
                end=datetime(y, 2, 28, 23, 0, 0),
                name=f"Epidemie_{y}",
            )
        )
        heatwaves.append(
            ScenarioWindow(
                start=datetime(y, 7, 10, 0, 0, 0),
                end=datetime(y, 7, 25, 23, 0, 0),
                name=f"Canicule_{y}",
            )
        )

        # Strikes occur at more "random" times, but remain reproducible (deterministic per year).
        # We generate two short windows per year: one in 1er semestre, un autre en 2e semestre.
        rng_1 = random.Random(10_000 + y)
        rng_2 = random.Random(20_000 + y)

        month_1 = rng_1.choice([1, 2, 3, 4, 5, 6])
        start_day_1 = rng_1.randint(3, 20)
        start_1 = datetime(y, month_1, start_day_1, 0, 0, 0)
        end_1 = start_1 + timedelta(days=4, hours=23)
        strikes.append(
            ScenarioWindow(
                start=start_1,
                end=end_1,
                name=f"Greve_{y}_A",
            )
        )

        month_2 = rng_2.choice([7, 8, 9, 10, 11])
        start_day_2 = rng_2.randint(3, 20)
        start_2 = datetime(y, month_2, start_day_2, 0, 0, 0)
        end_2 = start_2 + timedelta(days=4, hours=23)
        strikes.append(
            ScenarioWindow(
                start=start_2,
                end=end_2,
                name=f"Greve_{y}_B",
            )
        )

    return tuple(epidemics), tuple(heatwaves), tuple(strikes)


def _event_flags(
    ts: datetime,
    epidemic_windows: tuple[ScenarioWindow, ...],
    strike_windows: tuple[ScenarioWindow, ...],
    heatwave_windows: tuple[ScenarioWindow, ...],
) -> tuple[int, int, int, str]:
    e = 1 if any(w.contains(ts) for w in epidemic_windows) else 0
    s = 1 if any(w.contains(ts) for w in strike_windows) else 0
    h = 1 if any(w.contains(ts) for w in heatwave_windows) else 0
    parts: list[str] = []
    if e:
        parts.append("Epidemie")
    if s:
        parts.append("Greve")
    if h:
        parts.append("Canicule")
    return e, s, h, "+".join(parts) if parts else "Aucun"


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """
    Poisson sampling without numpy.
    - For small lambda: Knuth algorithm (exact).
    - For larger lambda: normal approximation (fast, good enough for synthetic data).
    """
    if lam <= 0:
        return 0
    if lam < 30.0:
        # Knuth: O(k) where k is the sample value; OK for small λ.
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= rng.random()
        return k - 1
    # Normal approximation with continuity correction
    x = rng.gauss(mu=lam, sigma=math.sqrt(lam))
    return max(0, int(round(x)))


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    return cov / math.sqrt(vx * vy)


def _weighted_choice(rng: random.Random, items: list[tuple[str, float]]) -> str:
    """Return one item given (value, weight) pairs."""
    total = sum(w for _, w in items)
    if total <= 0:
        # Fallback: uniform on values
        values = [v for v, _ in items] or ["Autre"]
        return rng.choice(values)
    r = rng.random() * total
    acc = 0.0
    for value, weight in items:
        acc += weight
        if r <= acc:
            return value
    return items[-1][0]


def _sample_motif_admission(
    rng: random.Random,
    service: ServiceName,
    month: int,
    dow_idx: int,
    hour: int,
    epi_flag: int,
    heat_flag: int,
) -> str:
    """
    Sample a main admission motif for the (service, time slot).
    Coarse but medically plausible, modulated by seasonality & scenarios.
    """
    is_winter = month in (12, 1, 2)
    is_night = hour < 8 or hour >= 22
    is_weekend = dow_idx >= 5

    if service == "Urgences":
        motifs = [
            ("Trauma", 3.0 + (1.0 if is_night or is_weekend else 0.0)),
            ("AVC", 1.0),
            ("Douleur thoracique", 1.4),
            ("Detresse respiratoire", 1.2 + (0.4 if epi_flag and is_winter else 0.0)),
            ("Grippe", 0.6 + (1.4 if epi_flag and is_winter else 0.0)),
            ("Autre", 2.0),
        ]
    elif service == "Cardio":
        motifs = [
            ("Douleur thoracique", 3.0),
            ("Infarctus", 1.8),
            ("Insuffisance cardiaque", 1.6),
            ("Troubles du rythme", 1.2),
            ("Autre", 1.0),
        ]
    elif service == "Neuro":
        motifs = [
            ("AVC", 3.0),
            ("Crise epileptique", 2.0),
            ("Trauma crânien", 1.6 + (0.5 if is_night else 0.0)),
            ("Syndrome confusionnel", 1.4),
            ("Autre", 1.0),
        ]
    elif service == "Infectieuses":
        flu_weight = 1.5 + (2.0 if (epi_flag and is_winter) else 0.0)
        motifs = [
            ("Grippe", flu_weight),
            ("Bronchiolite", 1.2 + (1.0 if (epi_flag and is_winter) else 0.0)),
            ("Pneumonie", 1.5),
            ("Gastro-enterite", 1.3),
            ("Sepsis", 1.0),
            ("Autre", 0.8),
        ]
    elif service == "Geriatrie":
        motifs = [
            ("Chute", 2.5 + (0.6 if is_winter else 0.0)),
            ("Deshydratation", 1.5 + (1.2 if heat_flag else 0.0)),
            ("Syndrome confusionnel", 1.8),
            ("Insuffisance cardiaque", 1.2),
            ("Infection urinaire", 1.0),
            ("Autre", 1.0),
        ]
    else:  # Pediatrie
        motifs = [
            ("Bronchiolite", 2.0 + (1.2 if (is_winter and epi_flag) else 0.0)),
            ("Asthme", 1.8),
            ("Trauma", 1.6 + (0.5 if is_weekend else 0.0)),
            ("Grippe", 1.5 + (0.8 if is_winter else 0.0)),
            ("Gastro-enterite", 1.3),
            ("Autre", 1.0),
        ]

    return _weighted_choice(rng, motifs)


def _sample_gravite(
    rng: random.Random,
    service: ServiceName,
    motif: str,
) -> int:
    """
    Triage severity score 1 (critique) to 5 (non urgent).
    Distribution biased by motif/type of service.
    """
    motif = motif.lower()
    if any(k in motif for k in ("avc", "infarctus", "sepsis", "detresse")):
        weights = [(1, 0.35), (2, 0.35), (3, 0.2), (4, 0.08), (5, 0.02)]
    elif "trauma" in motif or "trauma crânien" in motif or "trauma cr" in motif:
        weights = [(1, 0.10), (2, 0.25), (3, 0.35), (4, 0.20), (5, 0.10)]
    elif any(k in motif for k in ("grippe", "bronchiolite", "bronch", "pneumonie", "pneumo")):
        weights = [(1, 0.05), (2, 0.20), (3, 0.40), (4, 0.25), (5, 0.10)]
    elif any(k in motif for k in ("chute", "deshydrat", "déshydrat")):
        weights = [(1, 0.08), (2, 0.22), (3, 0.40), (4, 0.20), (5, 0.10)]
    else:
        # Default: mostly 3–4
        weights = [(1, 0.05), (2, 0.15), (3, 0.40), (4, 0.25), (5, 0.15)]

    # Map to list[(label_str, weight)] to reuse _weighted_choice
    label = _weighted_choice(rng, [(str(level), w) for level, w in weights])
    return int(label)


def _estimate_los_hours(service: ServiceName, gravite: int, motif: str) -> float:
    """
    Estimate an average LOS (length of stay) in hours for this group of patients.
    Used as a "micro" indicator; the bed occupancy model remains macro.
    """
    base_by_service: dict[ServiceName, float] = {
        "Urgences": 8.0,
        "Cardio": 96.0,
        "Neuro": 120.0,
        "Infectieuses": 72.0,
        "Geriatrie": 168.0,
        "Pediatrie": 48.0,
    }
    base = base_by_service.get(service, 48.0)
    factor_by_grav = {1: 2.0, 2: 1.6, 3: 1.0, 4: 0.5, 5: 0.25}
    factor = factor_by_grav.get(int(gravite), 1.0)

    motif_l = motif.lower()
    if any(k in motif_l for k in ("grippe", "bronchiolite", "gastro")):
        base *= 0.8
    if any(k in motif_l for k in ("avc", "infarctus", "sepsis")):
        base *= 1.3
    if any(k in motif_l for k in ("chute", "fracture")):
        base *= 1.1

    los = base * factor
    # Clamp to a reasonable range: [2h, 30 days]
    return max(2.0, min(los, 24.0 * 30.0))


def _infer_type_lit_requis(service: ServiceName, gravite: int, motif: str) -> str:
    """
    Map service + severity + motif to a coarse required bed type.
    Only 3 types as requested: Réanimation / Médecine interne / Chirurgie.
    """
    motif_l = motif.lower()
    if gravite in (1, 2):
        return "Reanimation"

    if any(k in motif_l for k in ("trauma", "fracture", "chir", "polytrauma")):
        return "Chirurgie"

    if service in ("Cardio", "Neuro"):
        if gravite <= 3:
            return "Reanimation"
        return "Medecine_interne"

    if service == "Infectieuses":
        return "Medecine_interne"

    if service == "Geriatrie":
        return "Medecine_interne"

    # Urgences / Pediatrie par défaut
    return "Medecine_interne"


def _facteur_externe_label(
    rng: random.Random,
    epi_flag: int,
    strike_flag: int,
    heat_flag: int,
) -> str:
    """
    External factor label: Normal / Canicule / Grève / Manif / Epidemie.
    - Epidemie renvoie 'Epidemie' (en plus de Type_Evenement).
    - Manif est un bruit rare hors scénarios.
    """
    if heat_flag:
        return "Canicule"
    if strike_flag:
        return "Greve"
    if epi_flag:
        return "Epidemie"
    # bruit rare
    if rng.random() < 0.01:
        return "Manif"
    return "Normal"


def generate_to_csv(cfg: GeneratorConfig, out_csv_path: str, run_checks: bool) -> dict[str, object]:
    """
    Generate the dataset and write it directly to CSV (streaming).
    Returns a small stats dict used for optional checks / logging.
    """
    rng = random.Random(cfg.seed)

    years = sorted({y for y in range(cfg.start.year, cfg.end.year + 1)})
    default_epi, default_heat, default_strike = _default_windows_for_years(years)
    epidemic_windows = cfg.epidemic_windows or default_epi
    heatwave_windows = cfg.heatwave_windows or default_heat
    strike_windows = cfg.strike_windows or default_strike

    # Base daily lambdas.
    #
    # - ED (Urgences) is anchored to 100k/year ≈ 274/day (given by spec).
    # - Other services are tuned to yield a realistic inpatient throughput for a ~1800-bed hospital,
    #   so that `Lits_Occupes` stays in a plausible 1200–1750-ish range (with variability).
    base_ed_daily = cfg.ed_passages_per_year / 365.0
    base_daily_by_service: dict[ServiceName, float] = {
        "Urgences": base_ed_daily,
        "Cardio": 70.0,
        "Neuro": 60.0,
        "Infectieuses": 55.0,
        "Geriatrie": 65.0,
        "Pediatrie": 75.0,
    }
    peak_hour: dict[ServiceName, int] = {
        "Urgences": 18,
        "Cardio": 11,
        "Neuro": 12,
        "Infectieuses": 14,
        "Geriatrie": 15,
        "Pediatrie": 19,
    }
    inpatient_rate: dict[ServiceName, float] = {
        "Urgences": 0.18,
        "Cardio": 0.55,
        "Neuro": 0.50,
        "Infectieuses": 0.65,
        "Geriatrie": 0.70,
        "Pediatrie": 0.40,
    }

    # Staff model baselines
    base_staff = 950.0
    staff_noise_sigma = 18.0

    # Beds model
    # LOS governs turnover; using a higher baseline yields realistic occupancy for 1800 beds.
    avg_los_hours_base = 168.0  # ~7 days
    beds_occupied = int(min(cfg.bed_capacity * 0.88, cfg.bed_capacity))  # start high (common in large hospitals)

    # EPI stock model
    epi_stock = cfg.epi_initial_stock
    pending_emergency_deliveries: dict[datetime, int] = {}

    # Optional checks accumulators (timestamp level)
    if run_checks:
        urg_adm_by_ts: list[int] = []
        infect_adm_by_ts: list[int] = []
        ger_adm_by_ts: list[int] = []
        epi_flag_by_ts: list[int] = []
        strike_flag_by_ts: list[int] = []
        staff_by_ts: list[int] = []
        rupture_by_ts: list[int] = []
        total_adm_by_ts: list[int] = []
        epi_cons_by_ts: list[int] = []
        occ_by_ts: list[int] = []
        month_by_ts: list[int] = []

    rows_written = 0

    # Keep only ONE datetime column (requested): `timestamp_admission`.
    # Everything else can be derived in BI (date/hour/day-of-week).
    header = [
        "timestamp_admission",
        "Service",
        "Nombre_Admissions",
        "motif_admission",
        "gravite",
        "duree_sejour_estimee",
        "type_lit_requis",
        "facteur_externe",
        "Lits_Occupes",
        "Lits_Disponibles",
        "Personnel_Present",
        "Stock_EPI",
        "EPI_Consommation",
        "Rupture_Stock",
        "Indicateur_Epidemie",
        "Type_Evenement",
        "Indicateur_Greve",
        "Indicateur_Canicule",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for ts in _date_range_inclusive(cfg.start, cfg.end, cfg.step):
            dow_idx = ts.weekday()  # Monday=0
            day_of_year = ts.timetuple().tm_yday
            seasonal = _seasonal_multiplier(day_of_year, cfg.seasonal_amplitude, cfg.seasonal_phase_days)

            epi, strike, heat, type_evt = _event_flags(
                ts,
                epidemic_windows=epidemic_windows,
                strike_windows=strike_windows,
                heatwave_windows=heatwave_windows,
            )

            # Generate admissions per service for this timestamp
            admissions_by_service: dict[ServiceName, int] = {}
            inpatient_adm = 0

            for service, base_daily in base_daily_by_service.items():
                base_hourly = base_daily / 24.0
                lam = base_hourly
                lam *= seasonal
                lam *= _dow_multiplier(dow_idx, service)
                lam *= _hourly_pattern(ts.hour, peak_hour[service])

                # Scenario injections
                if service == "Infectieuses" and epi == 1:
                    lam *= cfg.epidemic_multiplier_infectious
                if service == "Geriatrie" and heat == 1:
                    lam *= cfg.heatwave_multiplier_geriatrics

                adm = _poisson_sample(rng, lam)
                admissions_by_service[service] = adm
                inpatient_adm += int(round(adm * inpatient_rate[service]))

            total_adm = sum(admissions_by_service.values())

            # Personnel present (hospital-wide)
            shift_mult = 1.10 if 8 <= ts.hour < 20 else 0.90
            weekend_mult = 0.92 if dow_idx >= 5 else 1.00
            summer_mult = 0.85 if ts.month == 8 else 1.00
            strike_mult = cfg.strike_staff_multiplier if strike == 1 else 1.0
            staff_mean = base_staff * shift_mult * weekend_mult * summer_mult * strike_mult
            personnel_present = max(0, int(round(rng.gauss(mu=staff_mean, sigma=staff_noise_sigma))))

            # Beds occupied: discharges proportional to current occupancy
            # LOS increases during winter/epidemic/heatwave (cases more sévères), decreases un peu en été.
            is_winter = ts.month in (12, 1, 2)
            is_summer = ts.month in (7, 8)
            los_mult = 1.0
            if is_winter:
                los_mult *= 1.08
            if epi == 1:
                los_mult *= 1.15
            if heat == 1:
                los_mult *= 1.10
            if is_summer:
                los_mult *= 0.95
            avg_los_hours = avg_los_hours_base * los_mult

            exp_discharges = max(beds_occupied / avg_los_hours, 0.0)
            # When occupancy is very high, the system tends to accelerate discharges/transfers.
            pressure = max(0.0, (beds_occupied - 0.90 * cfg.bed_capacity) / cfg.bed_capacity)
            exp_discharges *= 1.0 + 0.9 * pressure
            discharges = _poisson_sample(rng, exp_discharges)
            beds_occupied = max(0, min(cfg.bed_capacity, beds_occupied + inpatient_adm - discharges))
            beds_available = cfg.bed_capacity - beds_occupied

            # EPI: deliveries + consumption
            # More frequent baseline re-supply: split daily volume on three rounds (05:00, 13:00, 21:00).
            if ts.hour in (5, 13, 21):
                epi_stock += cfg.epi_daily_delivery // 3
            if ts in pending_emergency_deliveries:
                epi_stock += pending_emergency_deliveries.pop(ts)

            crisis_multiplier = 1.0
            if epi == 1:
                crisis_multiplier *= 1.25
            if heat == 1:
                crisis_multiplier *= 1.08

            epi_consumption = max(0, int(round(crisis_multiplier * 6.0 * total_adm + 0.35 * personnel_present)))
            epi_stock -= epi_consumption
            rupture = 0
            if epi_stock <= 0:
                rupture = 1
                epi_stock = 0

            if epi_stock < cfg.epi_emergency_threshold:
                eta = ts + timedelta(hours=cfg.epi_emergency_lead_hours)
                if eta <= cfg.end:
                    pending_emergency_deliveries[eta] = pending_emergency_deliveries.get(eta, 0) + cfg.epi_emergency_delivery

            # Per-timeslot external factor & timestamp
            facteur_externe = _facteur_externe_label(rng, epi_flag=epi, strike_flag=strike, heat_flag=heat)
            # Emit one row per service
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            for service, adm in admissions_by_service.items():
                motif = _sample_motif_admission(
                    rng,
                    service=service,
                    month=ts.month,
                    dow_idx=dow_idx,
                    hour=ts.hour,
                    epi_flag=epi,
                    heat_flag=heat,
                )
                gravite = _sample_gravite(rng, service=service, motif=motif)
                los_hours = _estimate_los_hours(service=service, gravite=gravite, motif=motif)
                type_lit = _infer_type_lit_requis(service=service, gravite=gravite, motif=motif)

                writer.writerow(
                    {
                        "timestamp_admission": ts_str,
                        "Service": service,
                        "Nombre_Admissions": adm,
                        "motif_admission": motif,
                        "gravite": gravite,
                        "duree_sejour_estimee": round(los_hours, 1),
                        "type_lit_requis": type_lit,
                        "facteur_externe": facteur_externe,
                        "Lits_Occupes": beds_occupied,
                        "Lits_Disponibles": beds_available,
                        "Personnel_Present": personnel_present,
                        "Stock_EPI": epi_stock,
                        "EPI_Consommation": epi_consumption,
                        "Rupture_Stock": rupture,
                        "Indicateur_Epidemie": epi,
                        "Type_Evenement": type_evt,
                        "Indicateur_Greve": strike,
                        "Indicateur_Canicule": heat,
                    }
                )
                rows_written += 1

            # Checks (timestamp level)
            if run_checks:
                urg_adm_by_ts.append(admissions_by_service["Urgences"])
                infect_adm_by_ts.append(admissions_by_service["Infectieuses"])
                ger_adm_by_ts.append(admissions_by_service["Geriatrie"])
                epi_flag_by_ts.append(epi)
                strike_flag_by_ts.append(strike)
                staff_by_ts.append(personnel_present)
                rupture_by_ts.append(rupture)
                total_adm_by_ts.append(total_adm)
                epi_cons_by_ts.append(epi_consumption)
                occ_by_ts.append(beds_occupied)
                month_by_ts.append(ts.month)

    stats: dict[str, object] = {"rows": rows_written}
    if run_checks:
        winters = [urg for urg, m in zip(urg_adm_by_ts, month_by_ts, strict=True) if m <= 2 or m == 12]
        summers = [urg for urg, m in zip(urg_adm_by_ts, month_by_ts, strict=True) if 6 <= m <= 8]
        infect_epi = [x for x, f in zip(infect_adm_by_ts, epi_flag_by_ts, strict=True) if f == 1]
        infect_no = [x for x, f in zip(infect_adm_by_ts, epi_flag_by_ts, strict=True) if f == 0]
        staff_strike = [x for x, f in zip(staff_by_ts, strike_flag_by_ts, strict=True) if f == 1]
        staff_no = [x for x, f in zip(staff_by_ts, strike_flag_by_ts, strict=True) if f == 0]

        stats.update(
            {
                "urg_winter_mean": statistics.fmean(winters) if winters else float("nan"),
                "urg_summer_mean": statistics.fmean(summers) if summers else float("nan"),
                "infect_epi_mean": statistics.fmean(infect_epi) if infect_epi else float("nan"),
                "infect_no_mean": statistics.fmean(infect_no) if infect_no else float("nan"),
                "staff_strike_mean": statistics.fmean(staff_strike) if staff_strike else float("nan"),
                "staff_no_mean": statistics.fmean(staff_no) if staff_no else float("nan"),
                "rupture_hours": sum(rupture_by_ts),
                "corr_adm_epi": _pearson_corr([float(x) for x in total_adm_by_ts], [float(y) for y in epi_cons_by_ts]),
                "corr_adm_occ": _pearson_corr([float(x) for x in total_adm_by_ts], [float(y) for y in occ_by_ts]),
            }
        )
    return stats


def try_export_parquet_from_csv(csv_path: str, parquet_path: str) -> bool:
    """
    Optional: convert CSV -> Parquet if pyarrow is available.
    This may be blocked in environments enforcing native-extension restrictions.
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.csv as pacsv  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return False

    table = pacsv.read_csv(csv_path)
    pq.write_table(table, parquet_path)
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a synthetic hospital dataset (flows/HR/EPI).")
    p.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--end", default="2025-12-31 23:00:00", help="End datetime (inclusive, capped to 2025-12-31)")
    p.add_argument("--freq", default="H", help="Frequency. Only 'H' supported in stdlib mode.")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out", default="hospital_synth", help="Output file prefix (no extension)")
    p.add_argument("--format", choices=["csv", "parquet", "both"], default="csv", help="Export format")
    p.add_argument("--run-checks", action="store_true", help="Print coherence checks")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.freq.upper() != "H":
        raise SystemExit("Only hourly frequency ('H') is supported in this no-deps version.")

    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start.")

    max_end = datetime(2025, 12, 31, 23, 0, 0)
    if end > max_end:
        print(f"Note: --end clamped to {max_end.strftime('%Y-%m-%d %H:%M:%S')} (requested realism).")
        end = max_end

    cfg = GeneratorConfig(start=start, end=end, step=timedelta(hours=1), seed=args.seed)

    out_csv = f"{args.out}.csv"
    stats = generate_to_csv(cfg, out_csv_path=out_csv, run_checks=bool(args.run_checks))

    parquet_ok = False
    if args.format in ("parquet", "both"):
        parquet_ok = try_export_parquet_from_csv(out_csv, f"{args.out}.parquet")

    if args.run_checks:
        print("\n=== Coherence checks ===")
        print(
            f"- Urgences winter mean vs summer mean (higher expected): "
            f"{stats.get('urg_winter_mean', float('nan')):.2f} vs {stats.get('urg_summer_mean', float('nan')):.2f}"
        )
        print(
            f"- Infectieuses mean during epidemic vs outside: "
            f"{stats.get('infect_epi_mean', float('nan')):.2f} vs {stats.get('infect_no_mean', float('nan')):.2f}"
        )
        print(
            f"- Personnel mean during strike vs outside: "
            f"{stats.get('staff_strike_mean', float('nan')):.1f} vs {stats.get('staff_no_mean', float('nan')):.1f}"
        )
        print(f"- Rupture_Stock hours (should be >0 in stress): {int(stats.get('rupture_hours', 0))}")
        print(f"- Corr(total admissions, EPI consumption) (should be positive): {stats.get('corr_adm_epi', float('nan')):.3f}")
        print(f"- Corr(total admissions, bed occupancy) (often positive): {stats.get('corr_adm_occ', float('nan')):.3f}")

    if args.format == "csv":
        print(f"\nExport OK: {out_csv} ({stats['rows']:,} rows)")
    elif args.format == "parquet":
        if parquet_ok:
            print(f"\nExport OK: {args.out}.parquet ({stats['rows']:,} rows)")
        else:
            print(f"\nCSV exported but Parquet conversion failed/blocked. CSV: {out_csv} ({stats['rows']:,} rows)")
            return 2
    else:  # both
        if parquet_ok:
            print(f"\nExport OK: {out_csv} + {args.out}.parquet ({stats['rows']:,} rows)")
        else:
            print(f"\nExport OK: {out_csv} ({stats['rows']:,} rows) — Parquet conversion failed/blocked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

