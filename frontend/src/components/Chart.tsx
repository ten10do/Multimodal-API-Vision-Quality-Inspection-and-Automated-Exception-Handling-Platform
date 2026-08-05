import { useEffect, useRef } from "react";
import * as echarts from "echarts";

export interface ChartSeries {
  name: string;
  type: "bar" | "line";
  data: Array<number | [string | number, number]>;
  color?: string;
  smooth?: boolean;
}

export function Chart({
  option,
  height = 260,
}: {
  option: {
    title?: string;
    xAxis?: string[];
    series: ChartSeries[];
    yLabel?: string;
  };
  height?: number;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);

  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current);
    chartRef.current = chart;
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const series = option.series.map((s) => ({
      name: s.name,
      type: s.type,
      data: s.data,
      smooth: s.smooth ?? true,
      itemStyle: s.color ? { color: s.color } : undefined,
      barMaxWidth: 28,
    }));
    chart.setOption(
      {
        animation: false,
        tooltip: { trigger: "axis" },
        legend: { top: 0, textStyle: { color: "#94a3b8" } },
        grid: { left: 48, right: 16, top: 36, bottom: 28 },
        xAxis: {
          type: "category",
          data: option.xAxis ?? [],
          axisLabel: { color: "#94a3b8" },
        },
        yAxis: {
          type: "value",
          name: option.yLabel,
          nameTextStyle: { color: "#94a3b8" },
          axisLabel: { color: "#94a3b8" },
          splitLine: { lineStyle: { color: "#1e293b" } },
        },
        series,
      },
      true,
    );
  }, [option]);

  return <div ref={ref} style={{ width: "100%", height }} className="chart" />;
}
