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
        <Menu.Item key='european'>
          <Link to='/european'>
            <Icon type='appstore-o' />European
          </Link>
        </Menu.Item>

        <Menu.Item key='setting'>
          <Link to='/setting'>
            <Icon type='setting' />American
          </Link>
        </Menu.Item>

        <SubMenu key='component' title={<span><Icon type='bars' /><span>Asian</span></span>}>
          <Menu.Item key='charts'>
            <Link to='/charts' style={{ color: '#999' }}>
              Charts
            </Link>
          </Menu.Item>
          <Menu.Item key='grid'>
            <Link to='/grid' style={{ color: '#999' }}>
              Grid
            </Link>
          </Menu.Item>
          <Menu.Item key='table'>
            <Link to='/table' style={{ color: '#999' }}>
              Table
            </Link>
          </Menu.Item>
        </SubMenu>

      </Menu>
    )
  }
}

export default Menus
