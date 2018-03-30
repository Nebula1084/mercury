import React from 'react';
import MercuryTable from "../../components/table";
import { Input, Icon, Row, Col, Button } from 'antd';
import { connect, select } from 'dva';

class VolatilityImplier extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>
                <MercuryTable
                    header="Volatility Implier"
                    namespace="volatility"
                    addStock={false}
                    columns={this.props.volatility.columns}
                    dataSource={this.props.volatility.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )
    }
}

export default connect(({ volatility }) => ({ volatility }))(VolatilityImplier);
