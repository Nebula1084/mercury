import React from 'react';
import { Table, Button, Icon, Row, Col, Form, FormItem, Popconfirm, Select, Switch } from 'antd';
import ImportButton from './ImportButton'
import NumericInput from './NumericInput'
import CsvParse from '@vtex/react-csv-parse';
import styles from './table.less';
import { toCamelCase, buildHeaders } from 'utils';

export default class MercuryTable extends React.Component {

    constructor(props) {
        super(props)

    }

    handleClosedForm(value, record) {
        this.update(value, record.key, 'closedForm')
        if (value) {
            if (record.useGpu)
                this.update(false, record.key, 'useGpu')
            if (record.controlVariate)
                this.update(false, record.key, 'controlVariate')
        }
    }

    handleUseGpu(value, record) {
        if (record.closedForm != true) {
            this.update(value, record.key, 'useGpu')
        }
    }

    handleControlVariate(value, record) {
        if (record.closedForm != true)
            this.update(value, record.key, 'controlVariate')
    }


    expandedRowRender() {
        let self = this

        return function (record) {

            return (
                <Form layout="inline">
                    {record.closedForm != null ?
                        <Form.Item label="Closed Form">
                            <Switch checked={record.closedForm} onChange={value => self.handleClosedForm(value, record)} />
                        </Form.Item>
                        : ""
                    }
                    {record.useGpu != null ?
                        <Form.Item label="Use GPU">
                            <Switch checked={record.useGpu} onChange={value => self.handleUseGpu(value, record)} />
                        </Form.Item>
                        : ""
                    }
                    {record.controlVariate != null ?
                        <Form.Item label="Control Variate">
                            <Switch checked={record.controlVariate} onChange={value => self.handleControlVariate(value, record)} />
                        </Form.Item>
                        : ""}
                    {record.conf != null && record.conf != -1 ?
                        <Form.Item label="Confidence">
                            {record.conf}
                        </Form.Item>
                        : ""
                    }
                    {record.time != null ?
                        <Form.Item label="Time">
                            {record.time.toFixed(2)}ms
                        </Form.Item>
                        : ""
                    }
                </Form>
            )
        }
    }

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
                render: (text, record) => (
                    record.editing
                        ? <NumericInput style={{ margin: '-5px 0' }} value={text} onChange={value => this.update(value, record.key, dataIndex)} />
                        : text
                )
            });
        }
        ret.push({
            title: 'Option Type',
            key: 'optionType',
            dataIndex: 'optionType',
            render: (text, record) => {
                if (record.editing)
                    return (
                        <Select defaultValue={record.optionType} style={{ width: 70 }} onChange={value => this.update(value, record.key, 'optionType')}>
                            <Select.Option value='1'>CALL</Select.Option>
                            <Select.Option value='2'>PUT</Select.Option>
                        </Select>
                    )
                else
                    if (text == '1')
                        return 'CALL'
                    else if (text == '2')
                        return 'PUT'
                    else
                        return 'INVALID'
            }
        })

        ret.push({
            title: 'Action',
            key: 'action',
            width: 200,
            render: (text, record) => {
                return (
                    <div className="editable-row-operations">
                        {
                            record.editing ?
                                <span>
                                    <a onClick={() => this.save(record.key)}>Save</a>
                                    <span className="ant-divider" />
                                    <Popconfirm title="Sure to cancel?" onConfirm={() => this.cancel(record.key)}>
                                        <a>Cancel</a>
                                    </Popconfirm>
                                </span>
                                : <span>
                                    <a onClick={() => this.edit(record.key)}>Edit</a>
                                    <span className="ant-divider" />
                                    <a onClick={() => this.price(record.key)}>Price</a>
                                </span>
                        }
                    </div>
                )
            }
        })

        ret.push({
            title: 'Price',
            key: 'price',
            dataIndex: 'price',
            width: 100
        })
        return ret;
    }

    load = (data) => {
        this.props.dispatch({ type: this.props.namespace + '/import', payload: data });
    }

    add = () => {
        let headers = buildHeaders(this.props.columns)
        let data = {}
        for (let header of headers) {
            data[header] = '0'
        }
        data['optionType'] = '1';
        if (this.props.closedForm)
            data['closedForm'] = true;
        if (this.props.useGpu)
            data['useGpu'] = false;
        if (this.props.controlVariate)
            data['controlVariate'] = false;
        this.props.dispatch({ type: this.props.namespace + '/add', payload: data });
    }

    alter = () => {
        this.props.dispatch({ type: this.props.namespace + '/alter' });
    }

    update = (value, index, column) => {
        this.props.dispatch({ type: this.props.namespace + '/update', payload: { value, index, column } });
    }

    edit = (index) => {
        this.props.dispatch({ type: this.props.namespace + '/edit', payload: index })
    }

    save = (index) => {
        this.props.dispatch({ type: this.props.namespace + '/save', payload: index })
    }

    cancel = (index) => {
        this.props.dispatch({ type: this.props.namespace + '/cancel', payload: index })
    }

    price = (index) => {
        this.props.dispatch({ type: this.props.namespace + '/price', payload: index })
    }

    componentDidMount() {
    }

    render() {
        const headers = buildHeaders(this.props.columns);

        return (
            <div>
                <Row className={styles.header}>
                    <Col span={24}>
                        <h1>{this.props.header}</h1>
                    </Col>
                </Row>

                <div className={styles['showcase-toolbar']}>
                    <Form layout="inline">
                        <Form.Item>
                            <CsvParse
                                fileHeaders={headers}
                                keys={headers}
                                separators={[',', ';']}
                                onDataUploaded={this.load}
                                render={onChange => <ImportButton onChange={onChange} />}
                            />
                        </Form.Item>

                        <Form.Item>
                            <Button onClick={this.add} className={styles['tool-button']}>
                                <Icon type="plus-circle-o" /> Option
              </Button>
                        </Form.Item>
                        {
                            this.props.addStock ?
                                <Form.Item>
                                    <Button onClick={this.alter} className={styles['tool-button']}>
                                        <Icon type="plus-circle-o" /> Stock
                  </Button>
                                </Form.Item>
                                :
                                ""
                        }
                    </Form>
                </div>

                <div className={styles['showcase-container']}>
                    <Table expandedRowRender={this.expandedRowRender()} columns={this.buildColumns(this.props.columns)} dataSource={this.props.dataSource} />
                </div>
            </div>
        );
    }
}