import config from './config'
import { post, del, put, patch, get } from './request'

function toCamelCase(str) {
    let ret = str.split(' ').join('');
    ret = ret[0].toLowerCase() + ret.slice(1);
    return ret;
}

function buildHeaders(columns) {
    let headers = []
    for (let column of columns) {
        headers.push(toCamelCase(column))
    }
    headers.push('optionType');
    return headers;
}

export { config, post, del, put, patch, get, toCamelCase, buildHeaders }
