import React from 'react';
import MercuryTable from "../../components/table";
import { Input, Icon, Row, Col, Button } from 'antd';
import { connect, select } from 'dva';

class ClosePricer extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>
                <MercuryTable
                    header="Closed-form Geometric Asian Pricer"
                    namespace='close'
                    addStock={false}
                    columns={this.props.close.columns}
                    dataSource={this.props.close.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )
    }
}

export default connect(({ close }) => ({ close }))(ClosePricer);
