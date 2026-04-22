from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guard_band_postselection import QuantizedGBSR


def build_reference_instance(
    *,
    m: int = 1,
    distance_km: float = 1.0,
    modulation_variance: float = 1.0,
    excess_noise: float = 0.005,
    coding_overhead: float = 0.0,
) -> QuantizedGBSR:
    transmittance = QuantizedGBSR.fiber_transmittance_from_distance(distance_km)
    return QuantizedGBSR(
        m=m,
        modulation_variance=modulation_variance,
        transmittance=transmittance,
        excess_noise=excess_noise,
        coding_overhead=coding_overhead,
    )


def evaluate_epsilon_sweep(
    gbsr: QuantizedGBSR,
    tau: np.ndarray,
    guards: np.ndarray,
    epsilon_phi_g_values: np.ndarray,
    *,
    outside_weight_W: float,
) -> dict[str, np.ndarray]:
    records = [
        gbsr.evaluate_quantised_key_efficiency_from_epsilon_phi_g(
            tau,
            guards,
            float(eps),
            outside_weight_W=outside_weight_W,
            uf_alphabet_mode="conservative",
            include_dimension_reduction_penalty=True,
        )
        for eps in epsilon_phi_g_values
    ]

    holevo_corr = np.array([row["holevo_continuity_per_channel_use"] for row in records], dtype=float)
    kappa_uf = np.array([row["kappa_uf"] for row in records], dtype=float)
    kappa_f = np.array([row["kappa_f"] for row in records], dtype=float)
    key_gaussian = np.array([row["gaussian_key_per_pulse"] for row in records], dtype=float)
    key_after_holevo = np.array(
        [gbsr.symbols_per_pulse * row["key_per_symbol_after_holevo_continuity"] for row in records],
        dtype=float,
    )
    key_final = np.array([row["final_key_per_pulse"] for row in records], dtype=float)

    if np.any(np.diff(holevo_corr) < -1e-9):
        raise RuntimeError("Holevo continuity correction should be non-decreasing in epsilon_phi_g.")

    if np.any(np.diff(key_final) > 1e-9):
        raise RuntimeError("Final key should be non-increasing in epsilon_phi_g for fixed channel settings.")

    return {
        "holevo_corr": holevo_corr,
        "kappa_uf": kappa_uf,
        "kappa_f": kappa_f,
        "key_gaussian": key_gaussian,
        "key_after_holevo": key_after_holevo,
        "key_final": key_final,
    }


def plot_results(
    epsilon_phi_g_values: np.ndarray,
    sweep: dict[str, np.ndarray],
    *,
    gbsr: GBSR,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epsilon_phi_g_values, 2.0 * sweep["kappa_uf"], label="2 kappa(eps_cq, |UF|)")
    axes[0].plot(epsilon_phi_g_values, 2.0 * sweep["kappa_f"], label="2 kappa(eps_cq, 2)")
    axes[0].plot(epsilon_phi_g_values, sweep["holevo_corr"], linestyle="--", label="Total continuity")
    axes[0].set_xlabel("epsilon_phi_g")
    axes[0].set_ylabel("bits / channel use")
    axes[0].set_title("Continuity Terms")
    axes[0].set_xscale("log")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epsilon_phi_g_values,
        sweep["key_gaussian"],
        color="#0072B2",
        linestyle="--",
        linewidth=2.6,
        label="Gaussian key",
    )
    axes[1].plot(
        epsilon_phi_g_values,
        sweep["key_after_holevo"],
        color="#D55E00",
        linestyle="-",
        linewidth=2.6,
        label="After Holevo continuity",
    )
    axes[1].plot(
        epsilon_phi_g_values,
        sweep["key_final"],
        color="#000000",
        linestyle="-.",
        linewidth=2.6,
        label="Final key",
    )
    axes[1].set_xlabel("epsilon_phi_g")
    axes[1].set_ylabel("bits / pulse")
    axes[1].set_title("Key vs epsilon_phi_g")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "epsilon_phi_g_key_sweep.png", dpi=220)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    eps_axis = np.logspace(-10, -5, 300)
    z_size = int(gbsr.number_of_intervals)
    uf_size = 2 * z_size
    ax2.plot(eps_axis, [gbsr.continuity_kappa(float(e), z_size) for e in eps_axis], label="kappa(e, |Z|)")
    ax2.plot(eps_axis, [gbsr.continuity_kappa(float(e), uf_size) for e in eps_axis], label="kappa(e, |UF|)")
    ax2.set_xlabel("epsilon")
    ax2.set_ylabel("kappa(epsilon, |.|)")
    ax2.set_title("kappa Growth")
    ax2.set_xscale("log")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "kappa_growth.png", dpi=220)
    plt.close(fig2)


def epsilon_phi_g_from_total_epsilon(epsilon_total: float) -> float:
    """
    Convert epsilon_total (treated as epsilon_cq at W=0) to epsilon_phi_g.

    Uses the stable form of:
        epsilon_pur = sqrt(2 * eps_phi_g - eps_phi_g^2),
    with epsilon_total = epsilon_pur when outside_weight_W = 0.
    """
    eps_total = float(np.clip(epsilon_total, 0.0, 1.0))
    if eps_total <= 0.0:
        return 0.0
    if eps_total >= 1.0:
        return 1.0

    root = np.sqrt(max(1.0 - (eps_total * eps_total), 0.0))
    return float((eps_total * eps_total) / (1.0 + root))


def optimise_distance_point(
    *,
    distance_km: float,
    epsilon_phi_g: float,
    m: int,
    excess_noise: float,
    coding_overhead: float,
    v_mod_bounds: tuple[float, float],
    p_pass_bounds: tuple[float, float],
    outside_weight_W: float,
    initial_guess: np.ndarray,
) -> dict[str, float | np.ndarray]:
    transmittance = QuantizedGBSR.fiber_transmittance_from_distance(float(distance_km))

    def evaluate_candidate(v_mod_candidate: float, p_pass_candidate: float) -> dict[str, float]:
        v_mod_checked = float(np.clip(v_mod_candidate, *v_mod_bounds))
        p_pass_checked = float(np.clip(p_pass_candidate, *p_pass_bounds))

        gbsr = QuantizedGBSR(
            m=m,
            modulation_variance=v_mod_checked,
            transmittance=transmittance,
            excess_noise=excess_noise,
            coding_overhead=coding_overhead,
        )
        tau_arr = gbsr.build_equiprobable_tau()
        g_arr = gbsr.generate_g_arr_from_p_pass(p_pass_checked, tau_arr)

        metrics = gbsr.evaluate_quantised_key_efficiency_from_epsilon_phi_g(
            tau_arr,
            g_arr,
            epsilon_phi_g,
            outside_weight_W=outside_weight_W,
            uf_alphabet_mode="conservative",
            include_dimension_reduction_penalty=True,
        )
        return {
            "metrics": metrics,
            "v_mod": v_mod_checked,
            "p_pass": p_pass_checked,
        }

    def negative_key_per_pulse(params: Sequence[float]) -> float:
        try:
            evaluation = evaluate_candidate(float(params[0]), float(params[1]))
        except Exception:
            return float(np.inf)

        key_value = float(evaluation["metrics"].get("final_key_per_pulse", np.nan))
        if not np.isfinite(key_value):
            return float(np.inf)
        return float(-key_value)

    bounds = [v_mod_bounds, p_pass_bounds]
    powell_result = minimize(
        negative_key_per_pulse,
        x0=np.asarray(initial_guess, dtype=float),
        method="Powell",
        bounds=bounds,
        options={"xtol": 1e-4, "ftol": 1e-9, "maxiter": 250},
    )

    polish_start = (
        powell_result.x
        if powell_result.success and np.all(np.isfinite(powell_result.x))
        else np.asarray(initial_guess, dtype=float)
    )
    polish_result = minimize(
        negative_key_per_pulse,
        x0=np.asarray(polish_start, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )

    if polish_result.success and np.isfinite(polish_result.fun):
        best_v_mod, best_p_pass = polish_result.x
    elif powell_result.success and np.isfinite(powell_result.fun):
        best_v_mod, best_p_pass = powell_result.x
    else:
        grid_v_mod = np.linspace(v_mod_bounds[0], v_mod_bounds[1], 25)
        grid_p_pass = np.linspace(p_pass_bounds[0], p_pass_bounds[1], 25)
        best_value = float(np.inf)
        best_params: tuple[float, float] | None = None
        for v_candidate in grid_v_mod:
            for p_candidate in grid_p_pass:
                value = negative_key_per_pulse((float(v_candidate), float(p_candidate)))
                if value < best_value:
                    best_value = value
                    best_params = (float(v_candidate), float(p_candidate))
        if best_params is None:
            raise RuntimeError(f"Failed to find feasible parameters at distance {distance_km:.2f} km")
        best_v_mod, best_p_pass = best_params

    evaluation = evaluate_candidate(float(best_v_mod), float(best_p_pass))
    metrics = evaluation["metrics"]

    final_key = max(float(metrics.get("final_key_per_pulse", 0.0)), 0.0)
    gaussian_key = max(float(metrics.get("gaussian_key_per_pulse", 0.0)), 0.0)

    return {
        "final_key_per_pulse": final_key,
        "gaussian_key_per_pulse": gaussian_key,
        "v_mod": float(evaluation["v_mod"]),
        "p_pass": float(evaluation["p_pass"]),
        "next_guess": np.array([float(evaluation["v_mod"]), float(evaluation["p_pass"])], dtype=float),
    }


def evaluate_distance_sweep_for_total_epsilon(
    *,
    distances_km: np.ndarray,
    epsilon_total: float,
    m: int,
    excess_noise: float,
    coding_overhead: float,
    v_mod_bounds: tuple[float, float],
    p_pass_bounds: tuple[float, float],
    outside_weight_W: float,
) -> dict[str, np.ndarray]:
    epsilon_phi_g = epsilon_phi_g_from_total_epsilon(float(epsilon_total))
    initial_guess = np.array([np.mean(v_mod_bounds), 0.8], dtype=float)

    final_key_per_pulse = np.zeros_like(distances_km, dtype=float)
    gaussian_key_per_pulse = np.zeros_like(distances_km, dtype=float)
    optimal_v_mod = np.zeros_like(distances_km, dtype=float)
    optimal_p_pass = np.zeros_like(distances_km, dtype=float)

    for idx, distance_km in enumerate(distances_km):
        point = optimise_distance_point(
            distance_km=float(distance_km),
            epsilon_phi_g=epsilon_phi_g,
            m=m,
            excess_noise=excess_noise,
            coding_overhead=coding_overhead,
            v_mod_bounds=v_mod_bounds,
            p_pass_bounds=p_pass_bounds,
            outside_weight_W=outside_weight_W,
            initial_guess=initial_guess,
        )
        final_key_per_pulse[idx] = float(point["final_key_per_pulse"])
        gaussian_key_per_pulse[idx] = float(point["gaussian_key_per_pulse"])
        optimal_v_mod[idx] = float(point["v_mod"])
        optimal_p_pass[idx] = float(point["p_pass"])
        initial_guess = np.asarray(point["next_guess"], dtype=float)

    return {
        "epsilon_total": np.array([float(epsilon_total)], dtype=float),
        "epsilon_phi_g": np.array([float(epsilon_phi_g)], dtype=float),
        "final_key_per_pulse": final_key_per_pulse,
        "gaussian_key_per_pulse": gaussian_key_per_pulse,
        "optimal_v_mod": optimal_v_mod,
        "optimal_p_pass": optimal_p_pass,
    }


def evaluate_total_epsilon_distance_sweeps(
    *,
    distances_km: np.ndarray,
    epsilon_total_values: Sequence[float],
    m: int,
    excess_noise: float,
    coding_overhead: float,
    v_mod_bounds: tuple[float, float],
    p_pass_bounds: tuple[float, float],
    outside_weight_W: float,
) -> dict[float, dict[str, np.ndarray]]:
    sweeps: dict[float, dict[str, np.ndarray]] = {}
    for eps_total in epsilon_total_values:
        print(f"Optimising over distance for epsilon_total={eps_total:.0e} ...")
        sweeps[float(eps_total)] = evaluate_distance_sweep_for_total_epsilon(
            distances_km=distances_km,
            epsilon_total=float(eps_total),
            m=m,
            excess_noise=excess_noise,
            coding_overhead=coding_overhead,
            v_mod_bounds=v_mod_bounds,
            p_pass_bounds=p_pass_bounds,
            outside_weight_W=outside_weight_W,
        )
    return sweeps


def plot_total_epsilon_distance_sweeps(
    *,
    distances_km: np.ndarray,
    sweeps: dict[float, dict[str, np.ndarray]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    ordered_epsilons = sorted(sweeps.keys())
    colours = ["#E41A1C", "#4DAF4A", "#377EB8", "#00CED1", "#CC00FF"]

    all_positive_values: list[np.ndarray] = []
    for idx, eps_total in enumerate(ordered_epsilons):
        key_arr = np.asarray(sweeps[eps_total]["final_key_per_pulse"], dtype=float)
        key_plot = np.where(key_arr > 0.0, key_arr, np.nan)
        all_positive_values.append(key_arr[key_arr > 0.0])
        if eps_total == 0.0:
            label = r"$\epsilon_{\mathrm{tot}} = 0$"
        else:
            exponent = int(np.round(np.log10(eps_total)))
            label = rf"$\epsilon_{{\mathrm{{tot}}}} = 10^{{{exponent}}}$"
        ax.plot(
            distances_km,
            key_plot,
            color=colours[idx % len(colours)],
            linewidth=2.2,
            label=label,
        )

    positive_concat = np.concatenate([arr for arr in all_positive_values if arr.size > 0]) if all_positive_values else np.array([], dtype=float)
    if positive_concat.size > 0:
        y_min = max(float(np.nanmin(positive_concat)) * 0.8, 1e-12)
        y_max = float(np.nanmax(positive_concat)) * 1.2
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Key per pulse (bits)")
    ax.set_title("m = 1 key per pulse with new epsilon continuity (W = 0)")
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(distances_km)), float(np.max(distances_km)))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "epsilon_total_distance_key_sweep.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def write_key_per_pulse_eps_tot_data(
    *,
    distances_km: np.ndarray,
    sweeps: dict[float, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    ordered_epsilons = sorted(sweeps.keys())
    headers = ["distance_km"] + [f"key_per_pulse_{eps:.0e}" for eps in ordered_epsilons]
    columns = [np.asarray(distances_km, dtype=float)]
    for eps in ordered_epsilons:
        columns.append(np.asarray(sweeps[eps]["final_key_per_pulse"], dtype=float))

    matrix = np.column_stack(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        matrix,
        delimiter="\t",
        header="\t".join(headers),
        fmt="%.8e",
        comments="",
    )


def main() -> None:
    gbsr = build_reference_instance()

    tau = gbsr.build_equiprobable_tau()
    guards = gbsr.generate_g_arr_from_p_pass(0.90, tau)

    epsilon_phi_g_values = np.logspace(-10, -5, 61)
    outside_weight_W = 1e-15

    sweep = evaluate_epsilon_sweep(
        gbsr,
        tau,
        guards,
        epsilon_phi_g_values,
        outside_weight_W=outside_weight_W,
    )

    output_dir = THIS_DIR / "plots"
    plot_results(epsilon_phi_g_values, sweep, gbsr=gbsr, output_dir=output_dir)

    distances_km = np.linspace(1.0, 60.0, 300)
    epsilon_total_values = [1e-8, 1e-6, 1e-4]
    distance_sweeps = evaluate_total_epsilon_distance_sweeps(
        distances_km=distances_km,
        epsilon_total_values=epsilon_total_values,
        m=1,
        excess_noise=0.001,
        coding_overhead=0.001,
        v_mod_bounds=(0.01, 10.0),
        p_pass_bounds=(0.01, 1.0),
        outside_weight_W=0.0,
    )
    distance_plot_path = plot_total_epsilon_distance_sweeps(
        distances_km=distances_km,
        sweeps=distance_sweeps,
        output_dir=output_dir,
    )
    dat_path = THIS_DIR / "generated" / "m1_key_per_pulse_eps_tot.dat"
    write_key_per_pulse_eps_tot_data(
        distances_km=distances_km,
        sweeps=distance_sweeps,
        output_path=dat_path,
    )

    print("Saved plots:")
    print(f"  {output_dir / 'epsilon_phi_g_key_sweep.png'}")
    print(f"  {output_dir / 'kappa_growth.png'}")
    print(f"  {distance_plot_path}")
    print(f"  {dat_path}")
    print()
    print("Sanity snapshot:")
    print(f"  Gaussian key (eps=0): {sweep['key_gaussian'][0]:.6f} bits/pulse")
    print(f"  Final key (eps=0):    {sweep['key_final'][0]:.6f} bits/pulse")
    print(f"  Final key (eps=max):  {sweep['key_final'][-1]:.6f} bits/pulse")
    print()
    print("Distance sweep snapshot (new epsilon, W=0):")
    for eps_total in sorted(distance_sweeps.keys()):
        key_arr = distance_sweeps[eps_total]["final_key_per_pulse"]
        positive = np.flatnonzero(key_arr > 0.0)
        cutoff_distance = distances_km[int(positive[-1])] if positive.size > 0 else np.nan
        print(
            f"  eps_total={eps_total:.0e}: max key={np.max(key_arr):.6e} bits/pulse, "
            f"last positive distance={cutoff_distance:.2f} km"
        )


if __name__ == "__main__":
    main()
