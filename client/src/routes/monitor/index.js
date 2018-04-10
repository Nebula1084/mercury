import React from 'react';
import PricingMonitor from '../../components/monitor'
import { connect, select } from 'dva';

class Monitor extends React.Component {

    constructor(props) {
        super(props);
        const { dispatch } = this.props;
        this.dispatch = dispatch;
    }

    render() {
        return (
            <div>
                <PricingMonitor
                    columns={this.props.geometric.columns}
                    prices={this.props.geometric.prices}
                    dataSource={[this.props.geometric.rows[this.props.location.query.index]]}
                />
            </div>
        )

    }
}

export default connect(({ geometric }) => ({ geometric }))(Monitor);
