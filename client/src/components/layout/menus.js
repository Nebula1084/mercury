import React from 'react'
import { Menu, Icon } from 'antd'
import { Link } from 'dva/router'

const SubMenu = Menu.SubMenu

class Menus extends React.Component {

  getMenuSelectedKey = (routes) => {
    if (routes === undefined) return ''
    let gn = '';
    for (let i = routes.length - 1; i >= 0; i--) {
      const obj = routes[i];
      if ('path' in obj) {
        gn = obj.path;
        break;
      }
    }
    return gn;
  }

  render() {
    const { sidebarFold, onMenuClick, routes } = this.props
    const menukey = this.getMenuSelectedKey(routes);

    return (
      <Menu mode={sidebarFold ? 'vertical' : 'inline'} theme='light' onClick={onMenuClick} selectedKeys={Array.of(menukey)}>
        <SubMenu key='european' title={<span><Icon type='bars' /><span>European</span></span>}>
          <Menu.Item key='price'>
            <Link to='/european' style={{ color: '#999' }}>
              Price
            </Link>
          </Menu.Item>
          <Menu.Item key='volatility'>
            <Link to='/volatility' style={{ color: '#999' }}>
              Volatility
            </Link>
          </Menu.Item>
        </SubMenu>

        <Menu.Item key='american'>
          <Link to='/american'>
            <Icon type='setting' />American
          </Link>
        </Menu.Item>

        <SubMenu key='asian' title={<span><Icon type='bars' /><span>Asian</span></span>}>
          <Menu.Item key='close'>
            <Link to='/close' style={{ color: '#999' }}>
              Close
            </Link>
          </Menu.Item>
          <Menu.Item key='geometric'>
            <Link to='/geometric' style={{ color: '#999' }}>
              Geometric
            </Link>
          </Menu.Item>
          <Menu.Item key='arithmetic'>
            <Link to='/arithmetic' style={{ color: '#999' }}>
              Arithmetic
            </Link>
          </Menu.Item>
        </SubMenu>

      </Menu>
    )
  }
}

export default Menus
