import React, {Component} from 'react';
import {notFound} from './styles';
import {link} from '../homepage/styles';
import {DatePicker, message} from 'antd';

export default class NotFound extends Component {
  constructor(props) {
    super(props);
    this.state = {
      date: '',
    };
  }

  handleChange(date) {
    message.info('Selected: ' + date.toString());
    this.setState({date});
  }

  render() {
    return (
      <div style={{width: 400, margin: '100px auto'}}>
        <DatePicker onChange={value => this.handleChange(value)}/>
        <div style={{marginTop: 20}}>Current time:{this.state.date.toString()}</div>
      </div>
    );
  }
}
