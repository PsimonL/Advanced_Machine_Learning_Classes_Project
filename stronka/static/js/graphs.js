const SelectGraph = document.querySelectorAll("form select"),
graphButton = document.querySelector("form .btn-index");
fromCurrency = document.querySelector(".from select"),
toCurrency = document.querySelector(".to select");

for (let i = 0; i < SelectGraph.length; i++) {
    for(let currency_code in city_list){
        let selected = i === 0 ? currency_code === "USD" ? "selected" : "" : currency_code === "PLN" ? "selected" : "";
        let optionTag = `<option value="${currency_code}" ${selected}>${currency_code}</option>`;
        SelectGraph[i].insertAdjacentHTML("beforeend", optionTag);
    }
}

var roots = []

window.addEventListener("load", () => {
    const charts = [
        { id: "chartdiv_CO", metric: "CO (ppm)" },
        { id: "chartdiv_NO2", metric: "NO2 (ppb)" },
        { id: "chartdiv_O3", metric: "O3 (ppm)" },
        { id: "chartdiv_PM25", metric: "PM2.5 (ug/m3 LC)" },
        { id: "chartdiv_SO2", metric: "SO2 (ppb)" }
    ];

    charts.forEach(chart => {
        var root = am5.Root.new(chart.id);
        roots.push(root);
        wykres(root, chart.metric);
    });
});

graphButton.addEventListener("click", e =>{
    e.preventDefault();
    zmiana()
});

function zmiana() {
    const TitleTxt = document.querySelector(".title-graph h4");
    TitleTxt.innerText = `Daily air quality prediction for ${fromCurrency.value}:`;

    roots.forEach(root => root.dispose());
    roots = [];

    const charts = [
        { id: "chartdiv_CO", metric: "CO (ppm)" },
        { id: "chartdiv_NO2", metric: "NO2 (ppb)" },
        { id: "chartdiv_O3", metric: "O3 (ppm)" },
        { id: "chartdiv_PM25", metric: "PM2.5 (ug/m3 LC)" },
        { id: "chartdiv_SO2", metric: "SO2 (ppb)" }
    ];

    charts.forEach(chart => {
        var root = am5.Root.new(chart.id);
        roots.push(root);
        wykres(root, chart.metric);
    });
}

var absoluteMin = new Date("2023-09-13").getTime();
var absoluteMax = new Date("2023-12-31").getTime();

function wykres (root, metric) {
    root.setThemes([am5themes_Animated.new(root)]);
    var title = root.container.children.push(am5.Label.new(root, {
        text: `${metric}`,
        fontSize: 24,
        fontWeight: "bold",
        textAlign: "center",
        x: am5.percent(50),
        y: -1,
        fill: am5.color("#38B6FF"),
        centerX: am5.percent(50),
        paddingBottom: 100,
    }));
    root.container.setAll({
        paddingTop: 20
    });
    var chart = root.container.children.push(am5xy.XYChart.new(root, {
        panX: false,
        panY: false,
        wheelX: "panX",
        wheelY: "zoomX",
        layout: root.verticalLayout,
        pinchZoomX: true
    }));

    chart.get("colors").set("step", 2);

    var volumeAxis = chart.yAxes.push(am5xy.ValueAxis.new(root, {
        renderer: am5xy.AxisRendererY.new(root, {
            inside: true
        }),
        height: am5.percent(30),
        layer: 5,
        numberFormat: "#a"
    }));

    volumeAxis.get("renderer").labels.template.setAll({
        centerY: am5.percent(100),
        maxPosition: 0.98
    });

    var dateAxis = chart.xAxes.push(am5xy.GaplessDateAxis.new(root, {
        maxDeviation: 1,
        baseInterval: {timeUnit: "day", count: 1},
        renderer: am5xy.AxisRendererX.new(root, {}),
        tooltip: am5.Tooltip.new(root, {})
    }));

    dateAxis.get("renderer").labels.template.setAll({
        minPosition: 0.01,
        maxPosition: 0.99
    });

    var color1 = chart.get("colors").getIndex(0);
    var color2 = chart.get("colors").getIndex(4);

    var volumeSeries = chart.series.push(am5xy.LineSeries.new(root, {
        name: "Real Value",
        clustered: false,
        fill: color1,
        stroke: color1,
        valueYField: "real",
        valueXField: "date",
        xAxis: dateAxis,
        yAxis: volumeAxis,
        legendValueText: "{valueY}",
        tooltip: am5.Tooltip.new(root, {
            labelText: "{valueY}"
        })
    }));

    var volumeSeries1 = chart.series.push(am5xy.LineSeries.new(root, {
        name: "Predicted Value",
        clustered: false,
        fill: color2,
        stroke: color2,
        valueYField: "predicted",
        valueXField: "date",
        xAxis: dateAxis,
        yAxis: volumeAxis,
        legendValueText: "{valueY}",
        tooltip: am5.Tooltip.new(root, {
            labelText: "{valueY}"
        })
    }));

    var volumeLegend = volumeAxis.axisHeader.children.push(
        am5.Legend.new(root, {
            useDefaultMarker: true
        })
    );
    volumeLegend.data.setAll([volumeSeries]);

    var volumeLegend1 = volumeAxis.axisHeader.children.push(
        am5.Legend.new(root, {
            useDefaultMarker: true
        })
    );
    volumeLegend1.data.setAll([volumeSeries1]);

    chart.leftAxesContainer.set("layout", root.verticalLayout);
    chart.set("cursor", am5xy.XYCursor.new(root, {}))
    var scrollbar = chart.set("scrollbarX", am5xy.XYChartScrollbar.new(root, {
        orientation: "horizontal",
        height: 50
    }));

    var sbDateAxis = scrollbar.chart.xAxes.push(am5xy.GaplessDateAxis.new(root, {
        baseInterval: {
            timeUnit: "day",
            count: 1
        },
        renderer: am5xy.AxisRendererX.new(root, {})
    }));

    var sbVolumeAxis = scrollbar.chart.yAxes.push(
        am5xy.ValueAxis.new(root, {
            renderer: am5xy.AxisRendererY.new(root, {})
        })
    );

    var sbSeries = scrollbar.chart.series.push(am5xy.LineSeries.new(root, {
        valueYField: "real",
        valueXField: "date",
        xAxis: sbDateAxis,
        yAxis: sbVolumeAxis
    }));

    sbSeries.fills.template.setAll({
        visible: true,
        fillOpacity: 0.3
    });

    var sbSeries1 = scrollbar.chart.series.push(am5xy.LineSeries.new(root, {
        valueYField: "predicted",
        valueXField: "date",
        xAxis: sbDateAxis,
        yAxis: sbVolumeAxis
    }));

    sbSeries1.fills.template.setAll({
        visible: true,
        fillOpacity: 0.3
    });

    function loadData(min, max, side) {
        var minu = new Date(min).toISOString().slice(0, 10);
        var maxu = new Date(max).toISOString().slice(0, 10);
        var city = fromCurrency.value;
        var url = `/api/data/live?city=${city}&metric=${metric}&start=${minu}&end=${maxu}`;
        am5.net.load(url).then(function (result) {
            var jsonLiveData = am5.JSONParser.parse(result.response);
            var data = [];
            for (const entry of jsonLiveData) {
                var pom = new Date(
                    entry.date.slice(0, 4),
                    parseInt(entry.date.slice(5, 7)) - 1,
                    parseInt(entry.date.slice(8, 10))
                ).getTime();
                data.push({
                    date: pom,
                    real: entry[metric],
                    predicted: entry[`predicted_${metric}`]
                });
            }
            console.log(result.response);
            console.log(data);

            data.sort((a, b) => a.date - b.date);

            var processor = am5.DataProcessor.new(root, {
                numericFields: ["date", "value"]
            });
            processor.processMany(data);
            var start = dateAxis.get("start");
            var end = dateAxis.get("end");

            var seriesFirst = {};
            var seriesLast = {};

            if (side == "none") {
                if (data.length > 0) {
                    dateAxis.set("min", min);
                    dateAxis.set("max", max);
                    dateAxis.setPrivate("min", min);
                    dateAxis.setPrivate("max", max);
                    volumeSeries.data.setAll(data);
                    volumeSeries1.data.setAll(data);
                    sbSeries.data.setAll(data);
                    sbSeries1.data.setAll(data);
                    dateAxis.zoom(0, 1, 0);
                }
            } else if (side == "left") {
                seriesFirst[volumeSeries.uid] = volumeSeries.data.getIndex(0).date;
                seriesFirst[sbSeries.uid] = sbSeries.data.getIndex(0).date;
                seriesFirst[volumeSeries1.uid] = volumeSeries1.data.getIndex(0).date;
                seriesFirst[sbSeries1.uid] = sbSeries1.data.getIndex(0).date;

                for (var i = data.length - 1; i >= 0; i--) {
                    var date = data[i].date;
                    if (seriesFirst[volumeSeries.uid] > date) {
                        volumeSeries.data.unshift(data[i]);
                    }
                    if (seriesFirst[sbSeries.uid] > date) {
                        sbSeries.data.unshift(data[i]);
                    }
                    if (seriesFirst[volumeSeries1.uid] > date) {
                        volumeSeries1.data.unshift(data[i]);
                    }
                    if (seriesFirst[sbSeries1.uid] > date) {
                        sbSeries1.data.unshift(data[i]);
                    }
                }

                min = Math.max(min, absoluteMin);
                dateAxis.set("min", min);
                dateAxis.setPrivate("min", min);
                dateAxis.set("start", 0);
                dateAxis.set("end", (end - start) / (1 - start));
            } else if (side == "right") {
                seriesLast[volumeSeries.uid] = volumeSeries.data.getIndex(volumeSeries.data.length - 1).date;
                seriesLast[sbSeries.uid] = sbSeries.data.getIndex(sbSeries.data.length - 1).date;
                seriesLast[volumeSeries1.uid] = volumeSeries1.data.getIndex(volumeSeries.data.length - 1).date;
                seriesLast[sbSeries1.uid] = sbSeries1.data.getIndex(sbSeries.data.length - 1).date;

                for (var i = 0; i < data.length; i++) {
                    var date = data[i].date;
                    if (seriesLast[volumeSeries.uid] < date) {
                        volumeSeries.data.push(data[i]);
                    }
                    if (seriesLast[sbSeries.uid] < date) {
                        sbSeries.data.push(data[i]);
                    }
                    if (seriesLast[volumeSeries1.uid] < date) {
                        volumeSeries1.data.push(data[i]);
                    }
                    if (seriesLast[sbSeries1.uid] < date) {
                        sbSeries1.data.push(data[i]);
                    }
                }
                max = Math.min(max, absoluteMax);
                dateAxis.set("max", max);
                dateAxis.setPrivate("max", max);

                dateAxis.set("start", start / end);
                dateAxis.set("end", 1);
            }
        });
    }

    function loadSomeData() {
        var start = dateAxis.get("start");
        var end = dateAxis.get("end");

        var selectionMin = Math.max(dateAxis.getPrivate("selectionMin"), absoluteMin);
        var selectionMax = Math.min(dateAxis.getPrivate("selectionMax"), absoluteMax);

        var min = dateAxis.getPrivate("min");
        var max = dateAxis.getPrivate("max");

        if (start < 0) {
            loadData(selectionMin, min, "left");
        }
        if (end > 1) {
            loadData(max, selectionMax, "right");
        }
    }

    chart.events.on("panended", function () {
        loadSomeData();
    });


    var wheelTimeout;
    chart.events.on("wheelended", function () {
        if (wheelTimeout) {
            wheelTimeout.dispose();
        }

        wheelTimeout = chart.setTimeout(function () {
            loadSomeData();
        }, 50);
    });

    var oneWeekBack = absoluteMax - am5.time.getDuration("day", 7);

    loadData(oneWeekBack, absoluteMax, "none");

    chart.appear(1000, 500);
};
