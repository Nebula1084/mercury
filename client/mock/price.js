const Mock = require('mockjs');
const config = require('../src/utils/config');
const { api } = config;

const price = Mock.mock({
    'data':
        {
            price: 123
        }
});

module.exports = {
    ['GET /api/price'](req, res) {
        let data = price.data;
        res.status(200).json({
            data: data
        });
    },
}
