import React from 'react';
import MercuryTable from "../../components/table";
import { Input, Icon, Row, Col, Button } from 'antd';
import styles from './european.less';
import CsvParse from '@vtex/react-csv-parse';
import { connect, select } from 'dva';

const columns = [{
    title: 'Stock Price',
    dataIndex: 'stockPrice',
    key: 'stockPrice',
}, {
    title: 'Volatility',
    dataIndex: 'volatility',
    key: 'volatility',
}, {
    title: 'Maturity',
    dataIndex: 'maturity',
    key: 'maturity',
}, {
    title: 'Strike',
    dataIndex: 'strike',
    key: 'strike',
}, {
    title: 'Interest',
    dataIndex: 'interest',
    key: 'interest',
}, {
    title: 'Repo',
    dataIndex: 'repo',
    key: 'repo',
}, {
    title: 'Action',
    key: 'action',
    width: 200,
    render: (text, record) => (
        <span>
            <a href="#">Edit</a>
            <span className="ant-divider" />
            <a href="#">Delete</a>
            <span className="ant-divider" />
            <a href="#" className="ant-dropdown-link">
                More <Icon type="down" />
            </a>
        </span>
    ),
}];
const headers = []

for (let column of columns) {
    if (column.dataIndex != undefined) {
        headers.push(column.dataIndex)
    }
}

class ImportButton extends React.Component {
    render() {
        return (
            <div >
                <input ref="fileInput" type="file" onChange={this.props.onChange} style={{ display: "none" }} />
                <Button onClick={() => {
                    this.refs.fileInput.value = "";
                    this.refs.fileInput.click()
                }}>
                    <Icon type="upload" /> Import
                </Button>
            </div >
        )
    }
}

class EuropeanPricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    handleData = data => {
        this.dispatch({ type: 'european/add', payload: data });
    }

    render() {
        return (
            <div>
                <Row className={styles.header}>
                    <Col span={24}>
                        <h1>European Pricer</h1>
                    </Col>
                </Row>
                <div className={styles['showcase-toolbar']}>
                    <CsvParse
                        fileHeaders={headers}
                        keys={headers}
                        separators={[',', ';']}
                        onDataUploaded={this.handleData}
                        render={onChange => <ImportButton onChange={onChange} />}
                    />

                </div>
                <MercuryTable columns={columns} dataSource={this.props.european.rows} />
            </div>
        )
    }
}

export default connect(({ european }) => ({ european }))(EuropeanPricer);
