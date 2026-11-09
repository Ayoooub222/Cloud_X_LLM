"""Report figures: the plotted numbers match the cost model, and every figure renders."""
import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from src.cost.config import load_scenarios  # noqa: E402
from src.cost.measurements import ratio_value  # noqa: E402
from src.report import data, figures, make_figures, projection  # noqa: E402


@pytest.fixture(scope="module")
def report() -> dict:
    return data.cost_report()


def test_concurrency_series_have_three_runs_per_condition():
    for task, spec in data.SERIES.items():
        counts = data.concurrency_frame(task).groupby(["gpu", "concurrency"]).size()
        assert set(counts.index.get_level_values("gpu")) == {"rtx3070", "t4"}, task
        assert (counts >= 3).all(), task
        assert set(counts.index.get_level_values("concurrency")) == set(spec["levels"]), task


def test_ratio_at_saturation_is_the_cost_model_scale():
    target = load_scenarios().targets["g4dn.xlarge"]
    for task in ("audio", "vision"):
        plotted = data.ratio_frame(task)["ratio"].iloc[-1]
        assert plotted == pytest.approx(ratio_value(target.scale[task]["ratio"]), rel=1e-9), task


def test_sustained_rate_matches_the_cost_model(report):
    steps = {w["task"]: w["sustained_step"]["rate_rps"] for w in report["workloads"]}
    for task in ("audio", "vision"):
        assert figures.sustained_rate(data.load_frame(task)) == pytest.approx(steps[task]), task


def test_option_labels_and_colors():
    assert figures.option_label("g4dn.xlarge spot") == "g4dn.xlarge (T4), spot"
    assert figures.option_label("local-3070") == "RTX 3070 Laptop, owned"
    assert figures.option_color("g5.xlarge on-demand") != figures.option_color("g4dn.xlarge on-demand")


def test_every_figure_renders(tmp_path):
    assert make_figures.main(["--out", str(tmp_path)]) == 0
    written = sorted(p.stem for p in tmp_path.glob("*.png"))
    assert written == sorted(figures.all_figures())
    assert all(p.stat().st_size > 10_000 for p in tmp_path.glob("*.png"))


def test_unknown_figure_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        make_figures.main(["--out", str(tmp_path), "--only", "nope"])


def test_projection_reproduces_the_measured_point(report):
    """At the measured size and full capacity, the projection gives the cost model's own numbers."""
    options = {o.name: o for o in projection.vision_options(report)}
    vision = next(w for w in report["workloads"] if w["task"] == "vision")
    rows = {o["option"]: o for o in vision["options"]}
    spot = options["g5.xlarge spot"]
    full = spot.requests_per_hour * projection.HOURS_PER_MONTH
    expected = rows["g5.xlarge spot"]["eur_per_1000"]["1.0"]
    assert projection.option_cost_per_1000(spot, 2.0, full) == pytest.approx(expected)
    api = projection.api_eur_per_1000(report, batch=False)
    break_even = rows["g5.xlarge spot"]["break_even_utilization"]["gemini-3.1-flash-lite"] * full
    assert projection.best_cell([spot], 2.0, break_even * 1.01, api, None).winner == "g5.xlarge spot"
    assert projection.best_cell([spot], 2.0, break_even * 0.99, api, None).winner == projection.API


def test_projection_feasibility_rules(report):
    options = {o.name: o for o in projection.vision_options(report)}
    slo = projection.slo_s(report)
    assert projection.feasible(options["g4dn.xlarge spot"], 2.0, slo) == projection.INFEASIBLE_SLO
    assert projection.feasible(options["g4dn.xlarge spot"], 2.0, None) is None
    assert projection.feasible(options["g5.xlarge spot"], 2.0, slo) is None
    assert projection.feasible(options["g5.xlarge spot"], 10.0, None) == projection.INFEASIBLE_VRAM
    cell = projection.best_cell(list(options.values()), 10.0, 1e6, 1.0, None)
    assert cell.ratio is None and cell.winner == projection.INFEASIBLE_VRAM


def test_size_line_is_anchored_on_the_measurement(report):
    """At the measured model size the line is the cost model's full-load cost; at the break-even
    size it equals the API price."""
    for task in ("audio", "vision"):
        line = projection.cheapest_line(report, task)
        workload = next(w for w in report["workloads"] if w["task"] == task)
        row = next(o for o in workload["options"] if o["option"] == line.option)
        assert line.cost(line.anchor_b) == pytest.approx(row["eur_per_1000"]["1.0"])
        for price in workload["apis_eur_per_1000"].values():
            size = line.break_even_b(price)
            assert line.cost(size) == pytest.approx(price)
            assert line.break_even_b(price, 0.25) == pytest.approx(size / 4)
