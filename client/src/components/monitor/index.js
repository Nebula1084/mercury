import React from 'react';
import styles from './monitor.less';
import { Table } from 'antd';
import { toCamelCase } from 'utils'
import ReactEcharts from 'echarts-for-react';

export default class PricingMonitor extends React.Component {

    buildColumns(columns) {
        let ret = [];
        for (let column of columns) {
            let title = column;
            let dataIndex = toCamelCase(column);
            let key = dataIndex;
            ret.push({
                title: title,
                dataIndex: dataIndex,
                key: key,
            });
        }
        return ret;
    }
    render() {
        const data = this.props.prices;
        const reStyle = {
            width: '100%',
            height: '220px'
        }
        const base = 0;
        const sdoption = {
            title: {
                text: 'Confidence Band',
                subtext: 'Monte-Carlo',
                left: 'center'
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross',
                    animation: false,
                    label: {
                        backgroundColor: '#ccc',
                        borderColor: '#aaa',
                        borderWidth: 1,
                        shadowBlur: 0,
                        shadowOffsetX: 0,
                        shadowOffsetY: 0,
                        textStyle: {
                            color: '#222'
                        }
                    }
                },
                formatter: function (params) {
                    return params[2].name + '<br />' + params[2].value;
                }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: data.map(function (item) {
                    return item.time;
                }),
                axisLabel: {
                    formatter: function (value, idx) {
                        return value;
                    }
                },
                splitLine: {
                    show: false
                },
                boundaryGap: false
            },
            yAxis: {
                axisLabel: {
                    formatter: function (val) {
                        return val;
                    }
                },
                axisPointer: {
                    label: {
                        formatter: function (params) {
                            return params.value;
                        }
                    }
                },
                splitNumber: 3,
                splitLine: {
                    show: false
                }
            },
            series: [{
                name: 'L',
                type: 'line',
                data: data.map(function (item) {
                    return item.l + base;
                }),
                lineStyle: {
                    normal: {
                        opacity: 0
                    }
                },
                stack: 'confidence-band',
                symbol: 'none'
            }, {
                name: 'U',
                type: 'line',
                data: data.map(function (item) {
                    return item.u - item.l;
                }),
                lineStyle: {
                    normal: {
                        opacity: 0
                    }
                },
                areaStyle: {
                    normal: {
                        color: '#ccc'
                    }
                },
                stack: 'confidence-band',
                symbol: 'none'
            }, {
                type: 'line',
                data: data.map(function (item) {
                    return item.value + base;
                }),
                hoverAnimation: false,
                symbolSize: 6,
                itemStyle: {
                    normal: {
                        color: '#c23531'
                    }
                },
                showSymbol: false
            }]
        };

        return (
            <div>
                <div className={styles['showcase-container']}>
                    <Table pagination={false} columns={this.buildColumns(this.props.columns)} dataSource={this.props.dataSource} />
                </div>

                <ReactEcharts
                    option={sdoption}
                    style={reStyle} />
            </div>
        )
    }
}