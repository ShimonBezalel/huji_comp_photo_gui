
const FAST_STEP = 0.5;
const SLOW_STEP = 0.05;

const SERVER_HOST = "http://localhost:8000/";


function addEvent(element, eventName, callback) {
    if (element.addEventListener) {
        element.addEventListener(eventName, callback, false);
    } else if (element.attachEvent) {
        element.attachEvent("on" + eventName, callback);
    } else {
        element["on" + eventName] = callback;
    }
}

// b = document.querySelector("body");
// b.addEventListener()


addEvent(document, "keydown", function (e) {
    e = e || window["event"];
    if (e.key === 'Shift'){
        console.log("d shift");
        const sensitivity_controls = $("[id*=-step-]");
        const sliders = $("[id*=-slider-]");
        sensitivity_controls.each((index, elem) => elem.value = elem.value / 5);
        sliders.each((index, elem) => elem.step = elem.step / 5);
        // slider.step = SLOW_STEP;
    }
});

addEvent(document, "keyup", function (e) {
    e = e || window["event"];
    console.log("up key");

    if (e.key === "Shift"){
        const sensitivity_controls = $("[id*=-step-]");
        const sliders = $("[id*=-slider-]");
        sensitivity_controls.each((index, elem) => elem.value = elem.value * 5);
        sliders.each((index, elem) => elem.step = elem.step * 5);

    }


});

function setup_sliders() {
    const step_controllers = $("[id*=-step-]");
    step_controllers.attr("value", 0.01);
    step_controllers.change(function (e) {
        const slider = $("#" + e.target.id.replace("step", "slider"));
        slider.each((i, elem) => elem.step = e.target.value );

        const direct = $("#" + e.target.id.replace("step", "direct"));
        direct.each((i, elem) => elem.step = e.target.value );
    });

    const slider_controllers = $("[id*=-slider-]");
    slider_controllers.attr("step", 0.01);
    slider_controllers.attr("value", (slider_controllers.attr("max") - slider_controllers.attr("min")) / 2);

    slider_controllers.change(async function (e) {
        const direct = $("#" + e.target.id.replace("slider", "direct"));
        direct.each((i, elem) => elem.value = e.target.value);
        console.log(e.target.id);
        if (e.target.id.includes('viewpoint')){
            await request_stitch();
        }
        if (e.target.id.includes('focus')){
            await request_focus();
        }

    });

    const direct_controllers = $("[id*=-direct-]");
    direct_controllers.attr("step", 0.01);
    direct_controllers.each((i, elem) => {
        const slider = $("#" + elem.id.replace("direct", "slider"));
        elem.value = slider.val();
        const step = $("#" + elem.id.replace("direct", "step"));
        elem.step = step.val();
    } );
    direct_controllers.change(async function (e) {
        const slider = $("#" + e.target.id.replace("direct", "slider"));
        slider.each((i, elem) => elem.value = e.target.value);
        if (e.target.id.includes('viewpoint')){
            await request_stitch();
        }
        if (e.target.id.includes('focus')){
            await request_focus();
        }
    });
    const radius_inputs = $("[id*=-radius-input-]");
    radius_inputs.change(async function (e) {

        await request_focus();

    })


}

function setup_controllers() {
    const modal = $("#modal-file-upload");
    $("#file-button-upload").click(
        (e) => modal.fadeIn('fast')
    );
    $(".custom-file-input").on("change", function(e) {
        const fileNames = Array.from(e.target.files).map(f => f.name).join(", ");
      $(this).siblings(".custom-file-label").addClass("selected").html(fileNames);
    });
    $("#button-upload-save-files").click(
        async (e) => {
            e.target.innerHTML = '<span class="spinner-border spinner-border-sm"  id="button-upload-save-files-spinner" role="status" aria-hidden="true"></span> Loading...';
            await request_upload();
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
        (e) => request_motion()
    );
    // motion_button.tooltip();
    const dropdown_items = $(".dropdown-item");
    dropdown_items.click((e) => {
        console.log(e);

        const item_val = e.target.text;
        console.log(item_val);
        const menu_id = e.target.parentElement.id;
        console.log(menu_id);

        const input_elem = $("#" + menu_id.replace("dropdown", "input"));
        console.log(input_elem);

        input_elem.val(item_val);
        // e.target.parent.parent.siblings.val(item_val);
    })
}
setup_sliders();
setup_controllers();

async function request_motion() {
    const url = new URL(SERVER_HOST + 'motion/');

    const options = {
      method: 'GET'
    };
    const request = new Request(url.href, options);


    await fetch(request)
      .then(response => {
        if (response.status === 200) {
            return response.json();
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

async function request_stitch() {
    const shift_value = $('#viewpoint-direct-shift').val();
    const move_value = $('#viewpoint-direct-move').val();
    const stereo_value = $('#viewpoint-direct-stereo').val();

    const url = new URL(SERVER_HOST + 'viewpoint/');
    console.log(url.href);

    url.searchParams.append('shift', shift_value ? shift_value : 0.5);
    url.searchParams.append('move', move_value ? move_value : 0);
    url.searchParams.append('stereo', stereo_value? stereo_value : 0);
    console.log("Stitch/Viewpoint Request: "+ url.href);


    const canvas = $('#canvas-live-render');
    canvas.attr('src', url.href);

    // const options = {
    //   method: 'GET'
    // };
    // const request = new Request(url.href, options);
    //
    //
    // await fetch(request)
    //   .then(response => {
    //     if (response.status === 200) {
    //         return response.json();
    //     } else {
    //         console.error(response);
    //       throw new Error('Something went wrong on api server!');
    //     }
    //   }).then(json => json['body']['image']
    //     ).then(
    //         im => console.log(im)
    //         //todo: use image
    //     );

}

async function request_focus() {
    const focus_depth = $('#focus-direct-depth').val();
    const center = $('#focus-radius-input-center').val();
    const radius = $('#focus-radius-input-radius').val();


    const url = new URL(SERVER_HOST + 'focus/');
    url.searchParams.append('depth', focus_depth);
    url.searchParams.append('center', center);
    url.searchParams.append('radius', radius);
    console.log("Focus Request: "+ url.href);
    const canvas = $('#canvas-live-render');
    canvas.attr('src', url.href);

    // const request = new Request(url.href, options);
    //
    //
    // await fetch(request)
    //   .then(response => {
    //     if (response.status === 200) {
    //         return response.blob();
    //     } else {
    //         console.error(response);
    //       throw new Error('Something went wrong on api server!');
    //     }
    //   }).then(blobRes => {
    //       console.log(blobRes);
    //       var outputImg = document.createElement('img');
    //       outputImg.src = 'data:image/jpeg;'+blobRes;
    //   }
    //     );
}

async function request_upload() {
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

        } else {
            console.error(response);
          throw new Error('Something went wrong on api server!');
        }
      });


}

