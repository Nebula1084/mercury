import React from 'react';
import { connect } from 'dva';
import { Table, Button, Icon, Row, Col, Form, Popconfirm } from 'antd';
import ImportButton from './ImportButton'
import NumericInput from './NumericInput'
import CsvParse from '@vtex/react-csv-parse';
import styles from './table.less';

export default class MercuryTable extends React.Component {

  constructor(props) {
    super(props)

  }

  toCamelCase(str) {
    let ret = str.replace(' ', '');
    ret = ret[0].toLowerCase() + ret.slice(1);
    return ret;
  }

  buildHeaders(columns) {
    let headers = []
    for (let column of columns) {
      headers.push(this.toCamelCase(column))
    }
    return headers;
  }

  buildColumns(columns) {
    let ret = [];
    for (let column of columns) {
      let title = column;
      let dataIndex = this.toCamelCase(column)
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
                  <a>Price</a>
                </span>
            }
          </div>
        )
      }
    })
    return ret;
  }

  load = (data) => {
    this.props.dispatch({ type: 'european/import', payload: data });
  }

  add = () => {
    let headers = this.buildHeaders(this.props.columns)
    let data = {}
    for (let header of headers) {
      data[header] = '0'
    }
    this.props.dispatch({ type: 'european/add', payload: data });
  }

  update = (value, index, column) => {
    this.props.dispatch({ type: 'european/update', payload: { value, index, column } });
  }

  edit = (index) => {
    this.props.dispatch({ type: 'european/edit', payload: index })
  }

  save = (index) => {
    this.props.dispatch({ type: 'european/save', payload: index })
  }

  cancel = (index) => {
    this.props.dispatch({ type: 'european/cancel', payload: index })
  }

  componentDidMount() {
  }

  render() {
    const headers = this.buildHeaders(this.props.columns);

    return (
      <div>
        <Row className={styles.header}>
          <Col span={24}>
            <h1>European Pricer</h1>
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
                <Icon type="plus-circle-o" /> Add
              </Button>
            </Form.Item>
          </Form>
        </div>

        <div className={styles['showcase-container']}>
          <Table columns={this.buildColumns(this.props.columns)} dataSource={this.props.dataSource} />
        </div>
      </div>
    );
  }
}