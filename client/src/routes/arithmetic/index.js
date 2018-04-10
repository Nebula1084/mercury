import React from 'react';
import MercuryTable from "../../components/table";
import { Input, Icon, Row, Col, Button } from 'antd';
import { connect, select } from 'dva';

class ArithmeticPricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>
                <MercuryTable
                    header="Monte-Carlo Arithmetic Asian Pricer"
                    namespace="arithmetic"
                    addStock={true}
                    columns={this.props.arithmetic.columns}
                    dataSource={this.props.arithmetic.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )
    }
}

export default connect(({ arithmetic }) => ({ arithmetic }))(ArithmeticPricer);
