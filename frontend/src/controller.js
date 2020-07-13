/** Constants */
const debugging_host = "http://localhost:8000/";
const SERVER_HOST = debugging_host;


/** Reusable function definitions */
function addEvent(element, eventName, callback) {
    if (element.addEventListener) {
        element.addEventListener(eventName, callback, false);
    } else if (element.attachEvent) {
        element.attachEvent("on" + eventName, callback);
    } else {
        element["on" + eventName] = callback;
    }
}

function setup_sliders() {
    const step_controllers = $("[id*=-step-]");
    step_controllers.attr("value", 0.01);
    step_controllers.change(function (e) {
        const slider = $("#" + e.target.id.replace("step", "slider"));
        slider.each((i, elem) => elem.step = e.target.value);

        const direct = $("#" + e.target.id.replace("step", "direct"));
        direct.each((i, elem) => elem.step = e.target.value);
    });

    const slider_controllers = $("[id*=-slider-]");
    slider_controllers.attr("step", 0.01);
    slider_controllers.attr("value", (slider_controllers.attr("max") - slider_controllers.attr("min")) / 2);

    slider_controllers.change(async function (e) {
        const direct = $("#" + e.target.id.replace("slider", "direct"));
        direct.each((i, elem) => elem.value = e.target.value);
        console.log(e.target.id);
        if (e.target.id.includes('viewpoint')) {
            await request_stitch_handler();
        }
        if (e.target.id.includes('focus')) {
            await request_focus_handler();
        }

    });

    const direct_controllers = $("[id*=-direct-]");
    direct_controllers.attr("step", 0.01);
    direct_controllers.each((i, elem) => {
        const slider = $("#" + elem.id.replace("direct", "slider"));
        elem.value = slider.val();
        const step = $("#" + elem.id.replace("direct", "step"));
        elem.step = step.val();
    });
    direct_controllers.change(async function (e) {
        const slider = $("#" + e.target.id.replace("direct", "slider"));
        slider.each((i, elem) => elem.value = e.target.value);
        if (e.target.id.includes('viewpoint')) {
            await request_stitch_handler();
        }
        if (e.target.id.includes('focus')) {
            await request_focus_handler();
        }
    });
    const radius_inputs = $("[id*=-radius-input-]");
    radius_inputs.change(async function (e) {

        await request_focus_handler();

    });
    const slice_inputs = $("[id*=-slice-input-]");
    slice_inputs.change(async function (e) {
        await request_slice_handler();
    });


}

function setup_controllers() {
    const modal = $("#modal-file-upload");
    $("#file-button-upload").click(
        (e) => modal.fadeIn('fast')
    );
    // logic moved to button itself
    $("#file-button-save").attr('href', new URL(SERVER_HOST + 'save/').href);
    $("#motion-button-save").attr('href', new URL(SERVER_HOST + 'motion/').href);

    $(".custom-file-input").on("change", function (e) {
        const fileNames = Array.from(e.target.files).map(f => f.name).join(", ");
        $(this).siblings(".custom-file-label").addClass("selected").html(fileNames);
    });
    $("#button-upload-save-files").click(
        async (e) => {
            e.target.innerHTML = '<span class="spinner-border spinner-border-sm"  id="button-upload-save-files-spinner" role="status" aria-hidden="true"></span> Loading...';
            await request_upload_handler();
            e.target.innerHTML = "Save Changes";
            modal.fadeOut('slow');
        }
    );
    $("#button-upload-close").click(
        (e) => modal.fadeOut('fast')
    );
    $(".close").click(
        (e) => modal.fadeOut('fast')
    );
    const motion_button = $("#motion-button-calculate");
    motion_button.click(
        (e) => request_motion_handler()
    );
    // motion_button.tooltip();
    const dropdown_items = $(".dropdown-item");
    dropdown_items.click((e) => {
        const drop_down_values = {
            "focus-depth-select-direct-Micro": 0.1,
            "focus-depth-select-direct-Landscape": 0.9,
            "focus-shift-select-direct-Left Most Frame": 0,

            "viewpoint-shift-select-direct-Left Most Frame": 0,
            "viewpoint-shift-select-direct-Right Most Frame": 1,
            "viewpoint-shift-select-direct-Center Frame": 0.5,

            "viewpoint-move-select-direct-Zoom Out": 0.9,
            "viewpoint-move-select-direct-Zoom In": -0.9,
            "viewpoint-move-select-direct-Center": 0,

            "viewpoint-stereo-select-direct-Rotate Left": -0.9,
            "viewpoint-stereo-select-direct-Rotate Right": 0.9,
            "viewpoint-stereo-select-direct-Center": 0,
        };

        console.log(e);

        const item_val = e.target.text;
        const menu_id = e.target.parentElement.id;
        const input_elem = $("#" + menu_id.replace("dropdown", "input"));
        console.log(item_val)
        console.log(menu_id.replace("dropdown", "input"))
        console.log(input_elem)
        input_elem.val(item_val);
        const base = menu_id.replace("dropdown", "direct");
        var fields = base.split('-');
        const base_direct = fields[0] + "-" + fields[3] + "-" + fields[1];
        const direct_elem = $("#" + base_direct);
        direct_elem.val(drop_down_values[base + "-" + item_val]);
        direct_elem.change();
    })
}

async function request_motion_handler() {
    const url = new URL(SERVER_HOST + 'motion/');
    url.searchParams.append('add_string', true.toString());

    const options = {
        method: 'GET'
    };
    const request = new Request(url.href, options);


    await fetch(request)
        .then(response => {
            if (response.status === 200) {
                const j = response.json();
                console.log(j);
                return j;
            } else {
                console.error(response);
                throw new Error('Something went wrong on api server!');
            }
        }).then(json => json['as_string']
        ).then(
            vec_string => {
                const elem = $("#motion-vector-display");
                elem.val(vec_string);
                elem.tooltip('toggle');
            }
        );
}

async function request_slice_handler() {
    const frame_start = $('#viewpoint-slice-input-start-frame').val();
    const col_start = $('#viewpoint-slice-input-start-column').val();
    const frame_end = $('#viewpoint-slice-input-end-frame').val();
    const col_end = $('#viewpoint-slice-input-end-column').val();
    const slice = [[frame_start, col_start], [frame_end, col_end]].toString();

    const url = new URL(SERVER_HOST + 'viewpoint/');

    url.searchParams.append('slice', slice);

    console.log("Stitch/Slice Request: " + url.href);


    const canvas = $('#canvas-live-render');
    canvas.attr('src', url.href);

    const params_url = new URL(SERVER_HOST + 'slice_to_params/');

    params_url.searchParams.append('slice', slice);

    console.log("Params Request: "+ params_url.href);

    const request = new Request(params_url.href);
    const options = {
      method: 'GET'
    };

    await fetch(request, options)
      .then(response => {
        if (response.status === 200) {
            return response.json();
        } else {
            console.error(response);
          throw new Error('Something went wrong on api server!');
        }
      }).then( json => {
            const move = json['move'];
            const shift = json['shift'];
            const stereo =json['stereo'];

            $('#viewpoint-direct-shift').val(shift);
            $('#viewpoint-direct-move').val(move);
            $('#viewpoint-direct-stereo').val(stereo);

            $('#viewpoint-slider-shift').val(shift);
            $('#viewpoint-slider-move').val(move);
            $('#viewpoint-slider-stereo').val(stereo);
      });
}


async function request_stitch_handler() {
    const shift_value = $('#viewpoint-direct-shift').val();
    const move_value = $('#viewpoint-direct-move').val();
    const stereo_value = $('#viewpoint-direct-stereo').val();

    const viewpoint_url = new URL(SERVER_HOST + 'viewpoint/');

    viewpoint_url.searchParams.append('shift', shift_value ? shift_value : 0.5);
    viewpoint_url.searchParams.append('move', move_value ? move_value : 0);
    viewpoint_url.searchParams.append('stereo', stereo_value ? stereo_value : 0);
    console.log("Stitch/Viewpoint Request: " + viewpoint_url.href);


    const canvas = $('#canvas-live-render');
    canvas.attr('src', viewpoint_url.href);

    const slice_url = new URL(SERVER_HOST + 'params_to_slice/');

    slice_url.searchParams.append('shift', shift_value ? shift_value : 0.5);
    slice_url.searchParams.append('move', move_value ? move_value : 0);
    slice_url.searchParams.append('stereo', stereo_value ? stereo_value : 0);
    console.log("Slice Request: " + slice_url.href);

    const request = new Request(slice_url.href);
    const options = {
        method: 'GET'
    };

    await fetch(request, options)
        .then(response => {
            if (response.status === 200) {
                return response.json();
            } else {
                console.error(response);
                throw new Error('Something went wrong on api server!');
            }
        }).then(json => {
                const raw = json['slice'];
                const frame_start = raw[0];
                const col_start = raw[1];
                const frame_end = raw[2];
                const col_end = raw[3];
                $('#viewpoint-slice-input-start-frame').val(frame_start);
                $('#viewpoint-slice-input-start-column').val(col_start);
                $('#viewpoint-slice-input-end-frame').val(frame_end);
                $('#viewpoint-slice-input-end-column').val(col_end);
            }
        );


}

async function request_focus_handler() {
    const focus_depth = $('#focus-direct-depth').val();
    const center = $('#focus-radius-input-center').val();
    const radius = $('#focus-radius-input-radius').val();


    const url = new URL(SERVER_HOST + 'focus/');
    url.searchParams.append('depth', focus_depth);
    url.searchParams.append('center', center);
    url.searchParams.append('radius', radius);
    console.log("Focus Request: " + url.href);
    const canvas = $('#canvas-live-render');
    canvas.attr('src', url.href);

}

async function request_upload_handler() {
    const file = $("#customFile")[0].files[0];
    const formData = new FormData();

    const options = {
        method: 'POST',
        body: formData,
        // If you add this, upload won't work
        // headers: {
        //   'Content-Type': 'multipart/form-data',
        // }
    };


    formData.append('images', file);

    const request = new Request(SERVER_HOST + "upload_images/", options);
    await fetch(request)
        .then(response => {
            if (response.status === 200) {
                return response.json();

            } else {
                console.error(response);
                alert(response.text());
            }
        }).then((json) => {
                // const rows = json['rows'];
                const cols = json['cols'];
                // const channels = json['channels'];
                const frames = json['frames'];
                const input_center = $('#focus-radius-input-center');
                const input_rad = $('#focus-radius-input-radius');
                const half = Math.floor(frames / 2);

                input_center.attr('max', frames);
                input_center.val(half);
                input_rad.attr('max', half);
                input_rad.val(half);
                $('#viewpoint-slice-input-start-frame').attr('max', frames - 1);
                $('#viewpoint-slice-input-end-frame').attr('max', frames - 1);
                $('#viewpoint-slice-input-start-column').attr('max', cols - 1);
                $('#viewpoint-slice-input-end-column').attr('max', cols - 1);

                //todo edit presets
            }
        );


}

/** Page/logic setup on load */
addEvent(document, "keydown", function (e) {
    e = e || window["event"];
    if (e.key === 'Shift') {
        const sensitivity_controls = $("[id*=-step-]");
        const sliders = $("[id*=-slider-]");
        sensitivity_controls.each((index, elem) => elem.value = elem.value / 5);
        sliders.each((index, elem) => elem.step = elem.step / 5);
        // slider.step = SLOW_STEP;
    }
});
addEvent(document, "keyup", function (e) {
    e = e || window["event"];
    if (e.key === "Shift") {
        const sensitivity_controls = $("[id*=-step-]");
        const sliders = $("[id*=-slider-]");
        sensitivity_controls.each((index, elem) => elem.value = elem.value * 5);
        sliders.each((index, elem) => elem.step = elem.step * 5);

    }


});
setup_sliders();
setup_controllers();


