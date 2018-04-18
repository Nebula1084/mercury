import React from 'react';
import MercuryTable from "../../components/table";
import { Input, Icon, Row, Col, Button } from 'antd';
import { connect, select } from 'dva';

class ArithAsianPricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>
                <MercuryTable
                    header="Arithmetic Asian Pricer"
                    namespace="arithAsian"
                    closedForm={false}
                    useGpu={true}
                    controlVariate={true}
                    addStock={false}
                    columns={this.props.arithAsian.columns}
                    dataSource={this.props.arithAsian.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )
    }
}

export default connect(({ arithAsian }) => ({ arithAsian }))(ArithAsianPricer);
