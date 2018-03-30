import React from 'react';
import MercuryTable from "../../components/table";
import { Input, Icon, Row, Col, Button } from 'antd';
import styles from './european.less';
import { connect, select } from 'dva';
const columns = [
    'Stock Price', 'Volatility', 'Maturity', 'Strike', 'Interest', 'Repo',
]


class EuropeanPricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>                
                <MercuryTable columns={columns} dataSource={this.props.european.rows} dispatch={this.props.dispatch}/>
            </div>
        )
    }
}

export default connect(({ european }) => ({ european }))(EuropeanPricer);
