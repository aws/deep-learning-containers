var tfs_base_uri = '/tfs/v1/models/'
var custom_attributes_header = 'X-Amzn-SageMaker-Custom-Attributes'

function invocations(r) {
    var ct = (r.headersIn['Content-Type'] || '').split(';')[0].trim().toLowerCase()

    if (ct === 'application/json' || ct === 'application/jsonlines' || ct === 'application/jsons') {
        json_request(r)
    } else if (ct === 'text/csv') {
        csv_request(r)
    } else {
        return_error(r, 415, 'Unsupported Media Type: ' + (ct || 'Unknown'))
    }
}

function ping(r) {
    var uri = make_tfs_uri(r, false)
    if (uri === null) return

    function callback (reply) {
        if (reply.status == 200 && reply.responseText.includes('"AVAILABLE"')) {
            r.return(200)
        } else {
            r.error('failed ping' + reply.responseText)
            r.return(502)
        }
    }

    r.subrequest(uri, callback)
}

function ping_without_model(r) {
    // hack for TF 1.11 and MME
    // for TF 1.11, send an arbitrary fixed request to the default model.
    // if response is 400, the model is ok (but input was bad), so return 200
    // for MME, the default model name is None and does not exist
    // also return 200 in unlikely case our request was really valid

    var uri = make_tfs_uri(r, true)
    if (uri === null) return
    var options = {
        method: 'POST',
        body: '{"instances": "invalid"}'
    }

    function callback (reply) {
        if (reply.status == 200 || reply.status == 400 ||
        reply.responseText.includes('Servable not found for request: Latest(None)')) {
            r.return(200)
        } else {
            r.error('failed ping' + reply.responseText)
            r.return(502)
        }
    }

    r.subrequest(uri, options, callback)
}

function return_error(r, code, message) {
    r.headersOut['Content-Type'] = 'application/json'
    if (message) {
        r.return(code, JSON.stringify({error: String(message)}))
    } else {
        r.return(code)
    }
}

function tfs_json_request(r, json) {
    var uri = make_tfs_uri(r, true)
    if (uri === null) return
    var options = {
        method: 'POST',
        body: json
    }

    var accept = r.headersIn.Accept
    function callback (reply) {
        var body = reply.responseText
        if (reply.status == 400) {
            // "fix" broken json escaping in \'instances\' message
            body = body.replace("\\'instances\\'", "'instances'")
        }

        if (accept != undefined) {
            var content_types = accept.split(',').map(function(s) { return s.trim(); })
            if (content_types.indexOf('application/jsonlines') !== -1 || content_types.indexOf('application/json') !== -1) {
                body = body.replace(/\n/g, '')
                r.headersOut['Content-Type'] = 'application/json'
            }
        }
        r.return(reply.status, body)
    }

    r.subrequest(uri, options, callback)

}

var TFS_MODEL_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$/
var TFS_VERSION_RE = /^[0-9]+$/
var TFS_METHODS = ['predict', 'classify', 'regress']

function make_tfs_uri(r, with_method) {
    var attributes = parse_custom_attributes(r)
    var name = attributes['tfs-model-name']
    if (!TFS_MODEL_NAME_RE.test(name)) {
        return_error(r, 400, 'invalid tfs-model-name')
        return null
    }
    var uri = tfs_base_uri + name
    if ('tfs-model-version' in attributes) {
        if (!TFS_VERSION_RE.test(attributes['tfs-model-version'])) {
            return_error(r, 400, 'invalid tfs-model-version')
            return null
        }
        uri += '/versions/' + attributes['tfs-model-version']
    }
    if (with_method) {
        var m = attributes['tfs-method'] || 'predict'
        if (TFS_METHODS.indexOf(m) < 0) {
            return_error(r, 400, 'invalid tfs-method')
            return null
        }
        uri += ':' + m
    }
    return uri
}

function parse_custom_attributes(r) {
    var attributes = {}
    var kv_pattern = /tfs-[a-z\-]+=[^,]+/g
    var header = r.headersIn[custom_attributes_header]
    if (header) {
        var matches = header.match(kv_pattern)
        if (matches) {
            for (var i = 0; i < matches.length; i++) {
                var kv = matches[i].split('=')
                if (kv.length === 2) {
                    attributes[kv[0]] = kv[1]
                }
            }
        }
    }

    // for MME invocations, tfs-model-name is in the uri, or use default_tfs_model
    if (!attributes['tfs-model-name']) {
        var uri_pattern = /\/models\/[^,]+\/invoke/g
        var model_name = r.uri.match(uri_pattern)
        if (model_name && model_name[0]) {
            model_name = r.uri.replace('/models/', '').replace('/invoke', '')
            attributes['tfs-model-name'] = model_name
        } else {
            attributes['tfs-model-name'] = r.variables.default_tfs_model
        }
    }

    return attributes
}

function json_request(r) {
    var data = r.requestText

    if (is_tfs_json(data)) {
        tfs_json_request(r, data)
    } else if (is_json_lines(data)) {
        json_lines_request(r, data)
    } else {
        generic_json_request(r, data)
    }
}

function is_tfs_json(data) {
    return /"(instances|inputs|examples)"\s*:/.test(data)
}

function is_json_lines(data) {
    // objects separated only by (optional) whitespace means jsons/json-lines
    return /[}\]]\s*[\[{]/.test(data)
}

function generic_json_request(r, data) {
    if (! /^\s*\[\s*\[/.test(data)) {
        data = '[' + data + ']'
    }

    var json = '{"instances":' + data + '}'
    tfs_json_request(r, json)
}

function json_lines_request(r, data) {
    var lines = data.trim().split(/\r?\n/)
    var builder = []
    builder.push('{"instances":')
    if (lines.length != 1) {
        builder.push('[')
    }

    for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim()
        if (line) {
            var instance = (i == 0) ? '' : ','
            instance += line
            builder.push(instance)
        }
    }

    builder.push(lines.length == 1 ? '}' : ']}')
    tfs_json_request(r, builder.join(''))
}

function split_csv_fields(line) {
    var fields = [], cur = '', inq = false;
    for (var i = 0; i < line.length; i++) {
        var c = line[i];
        if (c === '"') {
            if (inq && i + 1 < line.length && line[i + 1] === '"') { cur += '"'; i++; }
            else { inq = !inq; }
        } else if (c === ',' && !inq) { fields.push(cur); cur = ''; }
        else { cur += c; }
    }
    fields.push(cur);
    return fields;
}

function csv_request(r) {
    var data = r.requestText
    var lines = data.trim().split(/\r?\n/)
    var builder = []
    builder.push('{"instances":[')

    for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim()
        if (line) {
            var fields = split_csv_fields(line)
            var has_multiple_columns = fields.length > 1

            var line_builder = []
            if (has_multiple_columns) {
                line_builder.push('[')
            }

            for (var j = 0; j < fields.length; j++) {
                if (j > 0) line_builder.push(',')
                var raw = fields[j]
                var probe = raw.trim()
                if (probe !== '' && /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(probe)) {
                    line_builder.push(probe)
                } else {
                    line_builder.push('"')
                    line_builder.push(raw.replace(/\\/g, '\\\\').replace(/"/g, '\\"'))
                    line_builder.push('"')
                }
            }

            if (has_multiple_columns) {
                line_builder.push(']')
            }

            var json_line = line_builder.join('')
            builder.push(json_line)

            if (i != lines.length - 1)
                builder.push(',')
        }
    }

    builder.push(']}')
    tfs_json_request(r, builder.join(''))
}

export default {invocations, ping, ping_without_model, return_error,
    tfs_json_request, make_tfs_uri, parse_custom_attributes,
    json_request, is_tfs_json, is_json_lines, generic_json_request,
    json_lines_request, csv_request};
