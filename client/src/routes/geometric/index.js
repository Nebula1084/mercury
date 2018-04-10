import React from 'react';
import MercuryTable from '../../components/table';
import PricingMonitor from '../../components/monitor'
import { Input, Icon, Row, Col, Button } from 'antd';
import { connect, select } from 'dva';

class GeometricPricer extends React.Component {

    constructor(props) {
        super(props);        
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {

        return (
            <div>
                <MercuryTable
                    header="Monte-Carlo Geometric Asian Pricer"
                    namespace="geometric"
                    addStock={true}
                    columns={this.props.geometric.columns}
                    dataSource={this.props.geometric.rows}
                    dispatch={this.props.dispatch}
                />
            </div>
        )

    }
}

export default connect(({geometric}) => ({ geometric }))(GeometricPricer);
