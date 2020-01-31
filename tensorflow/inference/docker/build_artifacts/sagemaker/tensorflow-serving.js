var tfs_base_uri = '/tfs/v1/models/'
var custom_attributes_header = 'X-Amzn-SageMaker-Custom-Attributes'

function invocations(r) {
    var ct = r.headersIn['Content-Type']

    if ('application/json' == ct || 'application/jsonlines' == ct || 'application/jsons' == ct) {
        json_request(r)
    } else if ('text/csv' == ct) {
        csv_request(r)
    } else {
        return_error(r, 415, 'Unsupported Media Type: ' + (ct || 'Unknown'))
    }
}

function ping(r) {
    var uri = make_tfs_uri(r, false)

    function callback (reply) {
        if (reply.status == 200 && reply.responseBody.includes('"AVAILABLE"')) {
            r.return(200)
        } else {
            r.error('failed ping' + reply.responseBody)
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
    var options = {
        method: 'POST',
        body: '{"instances": "invalid"}'
    }

    function callback (reply) {
        if (reply.status == 200 || reply.status == 400 ||
        reply.responseBody.includes('Servable not found for request: Latest(None)')) {
            r.return(200)
        } else {
            r.error('failed ping' + reply.responseBody)
            r.return(502)
        }
    }

    r.subrequest(uri, options, callback)
}

function return_error(r, code, message) {
    if (message) {
        r.return(code, '{"error": "' + message + '"}')
    } else {
        r.return(code)
    }
}

function tfs_json_request(r, json) {
    var uri = make_tfs_uri(r, true)
    var options = {
        method: 'POST',
        body: json
    }

    var accept = r.headersIn.Accept
    function callback (reply) {
        var body = reply.responseBody
        if (reply.status == 400) {
            // "fix" broken json escaping in \'instances\' message
            body = body.replace("\\'instances\\'", "'instances'")
        }

        if ('application/jsonlines' == accept || 'application/jsons' == accept) {
            body = body.replace(/\n/g, '')
            r.headersOut['Content-Type'] = accept
        }
        r.return(reply.status, body)
    }

    r.subrequest(uri, options, callback)

}

function make_tfs_uri(r, with_method) {
    var attributes = parse_custom_attributes(r)

    var uri = tfs_base_uri + attributes['tfs-model-name']
    if ('tfs-model-version' in attributes) {
        uri += '/versions/' + attributes['tfs-model-version']
    }

    if (with_method) {
        uri += ':' + (attributes['tfs-method'] || 'predict')
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
        if (model_name[0]) {
            model_name = r.uri.replace('/models/', '').replace('/invoke', '')
            attributes['tfs-model-name'] = model_name
        } else {
            attributes['tfs-model-name'] = r.variables.default_tfs_model
        }
    }

    return attributes
}

function json_request(r) {
    var data = r.requestBody

    if (is_json_lines(data)) {
        json_lines_request(r, data)
    } else if (is_tfs_json(data)) {
        tfs_json_request(r, data)
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

function csv_request(r) {
    var data = r.requestBody
    // look for initial quote or numeric-only data in 1st field
    var needs_quotes = data.search(/^\s*("|[\d.Ee+\-]+.*)/) != 0
    var lines = data.trim().split(/\r?\n/)
    var builder = []
    builder.push('{"instances":[')

    for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim()
        if (line) {
            var line_builder = []
            // Only wrap line in brackets if there are multiple columns.
            // If there's only one column and it has a string with a comma,
            // the input will be wrapped in an extra set of brackets.
            var has_multiple_columns = line.search(',') != -1

            if (has_multiple_columns) {
                line_builder.push('[')
            }

            if (needs_quotes) {
                line_builder.push('"')
                line_builder.push(line.replace('"', '\\"').replace(',', '","'))
                line_builder.push('"')
            } else {
                line_builder.push(line)
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
