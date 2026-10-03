"""Build factor tear-sheet options with pyecharts, without executable strings.

Only the two Custom-series geometry renderers are supplied by the fixed HTML
template. All series, values, fitted lines, axes and presentation options are
constructed and serialized through pyecharts' chart/component APIs here.
"""

import json
import math

import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar, Custom, HeatMap, Line, Scatter


COLORS = ["#6476c7", "#65a992", "#d59279", "#9382bd", "#c9ae6c", "#65a4c0", "#b384a1", "#8aa277"]
INK, GRID = "#52617a", "#edf0f6"


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _axis(name="", *, category=False, boundary_gap=True, scale=True, min_=None, max_=None, rotate=None):
    label = opts.LabelOpts(color="#8290a4", font_size=10, rotate=rotate)
    label.update(hideOverlap=True)
    return opts.AxisOpts(
        type_="category" if category else "value", name=name, name_location="middle", name_gap=46,
        is_scale=scale, boundary_gap=boundary_gap if category else None, min_=min_, max_=max_,
        name_textstyle_opts=opts.TextStyleOpts(color="#8793a5", font_size=10), axislabel_opts=label,
        axisline_opts=opts.AxisLineOpts(is_show=category,
                                       linestyle_opts=opts.LineStyleOpts(color="#dde4ef")),
        axistick_opts=opts.AxisTickOpts(is_show=False),
        splitline_opts=opts.SplitLineOpts(is_show=not category,
                                         linestyle_opts=opts.LineStyleOpts(color=GRID)),
        animation_opts=opts.AnimationOpts(animation=False),
    )


def _tooltip(trigger="axis", shadow=False):
    tooltip = opts.TooltipOpts(trigger=trigger, axis_pointer_type="shadow" if shadow else "line",
                               is_confine=True, textstyle_opts=opts.TextStyleOpts(font_size=11))
    tooltip.update(renderMode="richText")
    return tooltip


def _new(chart_class, chart_id, x_values, y_name="", *, zoom=False, legend=True,
         category=True, boundary_gap=True, min_=None, max_=None, scale=True, rotate=None,
         grid_bottom=None, tooltip_trigger="axis", shadow=False):
    chart = chart_class(init_opts=opts.InitOpts(
        bg_color="#fff", animation_opts=opts.AnimationOpts(animation=False),
        aria_opts=opts.AriaOpts(is_enable=True)))
    if isinstance(chart, Custom):
        # Custom has no built-in Cartesian axes; use the same pyecharts AxisOpts
        # components as RectChart, without loading an external custom plugin.
        chart.options.update(xAxis=[opts.AxisOpts().opts], yAxis=[opts.AxisOpts().opts])
    chart.add_xaxis(x_values)
    zoom_options = []
    if zoom:
        inside = opts.DataZoomOpts(type_="inside", filter_mode="none", range_start=0, range_end=100,
                                  is_zoom_on_mouse_wheel="ctrl")
        slider = opts.DataZoomOpts(type_="slider", filter_mode="none", range_start=0, range_end=100,
                                  pos_bottom=13, height=17, is_show_detail=False)
        slider.update(borderColor="#e3e8f1", brushSelect=False)
        zoom_options = [inside, slider]
    chart.set_global_opts(
        title_opts=opts.TitleOpts(is_show=False),
        legend_opts=opts.LegendOpts(is_show=legend, type_="scroll", pos_top=11, pos_left=18, pos_right=78,
                                   item_width=15, item_height=8,
                                   textstyle_opts=opts.TextStyleOpts(color="#7b889e", font_size=10)),
        tooltip_opts=_tooltip(tooltip_trigger, shadow),
        toolbox_opts=opts.ToolboxOpts(pos_left=None, pos_right=16, pos_top=9, item_size=13,
            feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="保存图像", name=f"qlib-{chart_id}",
                                                                pixel_ratio=2),
                restore=opts.ToolBoxFeatureRestoreOpts(title="恢复视图"), data_view=None,
                data_zoom=None, magic_type=None)),
        datazoom_opts=zoom_options,
        xaxis_opts=_axis(category=category, boundary_gap=boundary_gap, rotate=rotate),
        yaxis_opts=_axis(y_name, scale=scale, min_=min_, max_=max_),
    )
    chart.set_colors(COLORS.copy())
    chart.options.update(
        grid=opts.GridOpts(pos_left=68, pos_right=28, pos_top=57,
                           pos_bottom=(73 if zoom else 44) if grid_bottom is None else grid_bottom),
        textStyle=opts.TextStyleOpts(font_family="Segoe UI, Microsoft YaHei, sans-serif", color=INK),
    )
    return chart


def _dump(chart):
    # This is pyecharts' real option serialization, not a wrapper around an
    # independently assembled ECharts option dictionary. No JsCode is used.
    return json.loads(chart.dump_options_with_quotes())


def _line(chart_id, dates, series, y_name="", *, min_=None, max_=None):
    chart = _new(Line, chart_id, dates, y_name, zoom=len(dates) > 30, boundary_gap=False,
                 min_=min_, max_=max_)
    for index, item in enumerate(series):
        color = item.get("color", COLORS[index % len(COLORS)])
        chart.add_yaxis(
            item["name"], item["values"], is_connect_nones=False, is_symbol_show=len(dates) < 3,
            symbol_size=4, label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(width=item.get("width", 1.5), color=color,
                                               type_=item.get("type", "solid"), opacity=item.get("opacity", 1)),
            itemstyle_opts=opts.ItemStyleOpts(color=color),
            areastyle_opts=opts.AreaStyleOpts(opacity=.12 if item.get("area") else 0, color=color),
            emphasis_opts=opts.EmphasisOpts(focus="series"),
        )
    return _dump(chart)


def _distribution(horizon):
    items = horizon["quantiles"]
    chart = _new(Custom, "quantile-distribution", [f"Q{item['quantile']}" for item in items],
                 "期收益（bps）", legend=False, tooltip_trigger="item")
    bounds = []
    for index, item in enumerate(items):
        support = [point[0] for point in item.get("violin", []) if _finite(point[0]) and _finite(point[1])]
        box = item.get("box_bps")
        if box:
            support.extend([box[0], box[4]])
        bounds.append([index, min(support) if support else None, max(support) if support else None])
    chart.add("信号日期分布", render_item=None, dimensions=["quantile", "low", "high"],
              encode={"x": 0, "y": [1, 2]}, data=bounds,
              label_opts=opts.LabelOpts(is_show=False))
    return _dump(chart)


def _metric_options(horizon, metric, label):
    distribution = horizon["ic_distribution" if metric == "ic" else "rank_ic_distribution"]
    charts = {"ic-timeline": _line("ic-timeline", horizon["dates"], [
        {"name": label, "values": horizon["daily"][metric], "color": "#6476c7"},
        {"name": "滚动均值", "values": horizon["daily"][metric + "_rolling"],
         "color": "#65a992", "width": 2.2}], label, min_=-1, max_=1),
        "ic-cumulative": _line("ic-cumulative", horizon["dates"], [
            {"name": label + " 算术累计", "values": horizon["daily"][metric + "_cumulative"],
             "color": "#6476c7"}], label + " 算术累计")}
    histogram = _new(Custom, "ic-histogram", [], "日期数", category=False,
                     legend=False, min_=0, tooltip_trigger="item")
    histogram.options["xAxis"][0].update(_axis(label).opts)
    histogram.add("日期数", render_item=None, encode={"x": [0, 2, 3], "y": 1},
                  dimensions=["center", "count", "left", "right"],
                  data=[[item["center"], item["count"], item["center"] - item["width"] / 2,
                         item["center"] + item["width"] / 2] for item in distribution["histogram"]],
                  itemstyle_opts=opts.ItemStyleOpts(color="#7e91cb", opacity=.75),
                  label_opts=opts.LabelOpts(is_show=False))
    charts["ic-histogram"] = _dump(histogram)
    points = distribution["qq"]
    qq = _new(Scatter, "ic-qq", [], f"观测 {label}", category=False, legend=False, tooltip_trigger="item")
    qq.options["xAxis"][0].update(_axis("正态理论分位数").opts)
    qq.add_yaxis("观测分位数", points, symbol_size=4, label_opts=opts.LabelOpts(is_show=False),
                 itemstyle_opts=opts.ItemStyleOpts(color="#7b74c4", opacity=.75))
    if len(points) > 1:
        sample = np.asarray(points, dtype=float)
        centered_x = sample[:, 0] - sample[:, 0].mean()
        centered_y = sample[:, 1] - sample[:, 1].mean()
        variance = float(np.dot(centered_x, centered_x))
        slope = float(np.dot(centered_x, centered_y) / variance) if variance > 0 else 0.0
        fitted = [[float(x), float(sample[:, 1].mean() + slope * (x - sample[:, 0].mean()))]
                  for x in (sample[0, 0], sample[-1, 0])]
        reference = _new(Line, "ic-qq", [], category=False, legend=False)
        reference.add_yaxis("最小二乘参考线", [opts.LineItem(value=point) for point in fitted],
                            is_symbol_show=False, label_opts=opts.LabelOpts(is_show=False),
                            linestyle_opts=opts.LineStyleOpts(color="#65a992", type_="dashed", width=1.3))
        reference.options["series"][0]["silent"] = True
        qq.overlap(reference)
    charts["ic-qq"] = _dump(qq)
    monthly = distribution["monthly"]
    cells = [point for point in monthly["values"] if _finite(point[2])]
    span = max([.01, *[abs(point[2]) for point in cells]])
    heatmap = _new(HeatMap, "ic-monthly", [f"{month}月" for month in monthly["months"]],
                   legend=False, tooltip_trigger="item")
    years = [str(year) for year in monthly["years"]]
    heatmap.add_yaxis(f"月度平均 {label}", years, cells,
                      label_opts=opts.LabelOpts(is_show=len(years) < 8, font_size=10, color="#4b6056"),
                      itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=3))
    heatmap.options["yAxis"][0].update(_axis(category=True).opts)
    heatmap.options["visualMap"] = opts.VisualMapOpts(min_=-span, max_=span, is_calculable=True, orient="horizontal",
            pos_left="center", pos_bottom=13, item_width=9, item_height=145,
            range_color=["#dd827f", "#f7f8e9", "#b5cb82", "#4f977b"],
            textstyle_opts=opts.TextStyleOpts(color="#7e8a9f", font_size=10))
    heatmap.options["grid"] = opts.GridOpts(pos_left=64, pos_right=26, pos_top=32, pos_bottom=92)
    heatmap.options["yAxis"][0]["axisLabel"].update(interval=0)
    charts["ic-monthly"] = _dump(heatmap)
    sector = horizon["sector"]
    if sector.get("available"):
        overview = sector["overview"]
        zoom = len(overview) > 12
        bar = _new(Bar, "sector-ic", [item["name"] for item in overview], label, legend=False,
                   zoom=zoom, scale=False, rotate=30, grid_bottom=104 if zoom else 80)
        bar.add_yaxis(label, [item[metric + "_mean"] for item in overview], bar_max_width=33,
                      label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="#6476c7"),
                      emphasis_opts=opts.EmphasisOpts(focus="series"))
        charts["sector-ic"] = _dump(bar)
    return charts


def _mean_quantile_options(horizons, quantile_data):
    quantiles = [item["quantile"] for item in next(iter(quantile_data.values()))]
    bar = _new(Bar, "mean-quantile", [f"Q{group}" for group in quantiles], "期收益（bps）",
               scale=False, shadow=True)
    for horizon in horizons:
        groups = {item["quantile"]: item["mean_return_bps"] for item in quantile_data[str(horizon)]}
        bar.add_yaxis(f"{horizon} 日", [groups.get(group) for group in quantiles], bar_max_width=30,
                      label_opts=opts.LabelOpts(is_show=False), emphasis_opts=opts.EmphasisOpts(focus="series"))
    return _dump(bar)


def _return_options(horizon_name, dates, quantiles, sector_returns):
    """Serialize supplied gross or net return values, without recalculating fees."""
    common = {
        "quantile-distribution": _distribution({"quantiles": quantiles}),
        "quantile-cumulative": _line("quantile-cumulative", dates, [
            {"name": f"Q{item['quantile']}", "values": item["cumulative_bps"]} for item in quantiles], "累计期收益（bps）"),
    }
    for index, item in enumerate(sector_returns):
        sector = _new(Bar, f"sector-{index}", [f"Q{group['quantile']}" for group in item["groups"]],
                      "期收益（bps）", legend=False, scale=False)
        sector.add_yaxis(f"{horizon_name} 日期收益", [opts.BarItem(
            name=f"Q{group['quantile']}", value=group["mean_return_bps"],
            itemstyle_opts=opts.ItemStyleOpts(color=COLORS[(group["quantile"] - 1) % len(COLORS)]))
            for group in item["groups"]], bar_max_width=35, label_opts=opts.LabelOpts(is_show=False))
        common[f"sector-{index}"] = _dump(sector)
    return common


def _add_chart_options(payload):
    """Attach pyecharts-generated chart options, retaining every diagnostic value."""
    fee_keys = [item["value"] for item in payload["meta"].get("fee_options", []) if item["value"] != "none"]
    for factor in payload["factors"]:
        horizon_values = factor["horizons"]
        factor["chart_options"] = {"mean-quantile": _mean_quantile_options(payload["meta"]["horizons"],
            {name: horizon["quantiles"] for name, horizon in horizon_values.items()})}
        if fee_keys:
            factor["fee_chart_options"] = {key: {"mean-quantile": _mean_quantile_options(payload["meta"]["horizons"],
                {name: horizon["fee_scenarios"][key]["quantiles"] for name, horizon in horizon_values.items()})}
                for key in fee_keys}
        turnover = _line("quantile-turnover", factor["turnover"]["dates"], [
            {"name": "顶组", "values": factor["turnover"]["top"], "color": "#65a992", "area": True},
            {"name": "底组", "values": factor["turnover"]["bottom"], "color": "#6476c7", "area": True}],
            "换手率", min_=0, max_=1)
        autocorrelation = _line("factor-autocorrelation", factor["autocorrelation"]["dates"], [
            {"name": "因子自相关", "values": factor["autocorrelation"]["values"], "color": "#6476c7", "area": True}],
            "相关系数", min_=-1, max_=1)
        for horizon_name, horizon in horizon_values.items():
            dates = horizon["dates"]
            sector_returns = horizon["sector"]["quantile_returns"] if horizon["sector"].get("available") else []
            common = _return_options(horizon_name, dates, horizon["quantiles"], sector_returns)
            common.update({
                "coverage": _line("coverage", dates, [
                    {"name": "因子覆盖率", "values": horizon["daily"]["coverage"], "color": "#65a992"},
                    {"name": "因子与收益配对覆盖率", "values": horizon["daily"]["pair_coverage"], "color": "#6476c7"}], "覆盖率", min_=0, max_=1),
                "quantile-turnover": turnover, "factor-autocorrelation": autocorrelation,
            })
            if horizon["sector"].get("available"):
                overview = horizon["sector"]["overview"]
                zoom = len(overview) > 12
                overview_chart = _new(Bar, "sector-ic", [item["name"] for item in overview],
                    "相关系数", zoom=zoom, scale=False, rotate=30, grid_bottom=104 if zoom else 80)
                for metric, label in (("ic", "Pearson IC"), ("rank_ic", "Rank IC")):
                    overview_chart.add_yaxis(label, [item[metric + "_mean"] for item in overview],
                        bar_max_width=33, label_opts=opts.LabelOpts(is_show=False),
                        emphasis_opts=opts.EmphasisOpts(focus="series"))
                common["sector-ic"] = _dump(overview_chart)
            horizon["chart_options"] = {"common": common,
                "ic": _metric_options(horizon, "ic", "Pearson IC"),
                "rank_ic": _metric_options(horizon, "rank_ic", "Rank IC")}
            for key in fee_keys:
                scenario = horizon["fee_scenarios"][key]
                scenario["chart_options"] = {"common": _return_options(horizon_name, dates,
                    scenario["quantiles"], scenario["sector"]["quantile_returns"])}
    return payload
