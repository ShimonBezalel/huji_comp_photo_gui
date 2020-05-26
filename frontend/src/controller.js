// import {SENSITIVITY_VAL} from "./constants";
const FAST_STEP = 0.5;
const SLOW_STEP = 0.05;

const API_GATEWAY = "https://example.com";

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
        const sensitivity_controls = $("[id^=step-]");
        const sliders = $("[id^=slider-]");
        sensitivity_controls.each((index, elem) => elem.value = elem.value / 5);
        sliders.each((index, elem) => elem.step = elem.step / 5);
        // slider.step = SLOW_STEP;
    }
});

addEvent(document, "keyup", function (e) {
    e = e || window["event"];
    console.log("up key");

    if (e.key === "Shift"){
        const sensitivity_controls = $("[id^=step-]");
        const sliders = $("[id^=slider-]");
        sensitivity_controls.each((index, elem) => elem.value = elem.value * 5);
        sliders.each((index, elem) => elem.step = elem.step * 5);

    }


});

function setup_sliders() {
    const step_controllers = $("[id^=step-]");
    step_controllers.attr("value", 0.01);
    step_controllers.change(function (e) {
        const slider = $("#" + e.target.id.replace("step", "slider"));
        slider.each((i, elem) => elem.step = e.target.value );

        const direct = $("#" + e.target.id.replace("step", "direct"));
        direct.each((i, elem) => elem.step = e.target.value );
    });

    const slider_controllers = $("[id^=slider-]");
    slider_controllers.attr("step", 0.01);
    slider_controllers.attr("value", (slider_controllers.attr("max") - slider_controllers.attr("min")) / 2);

    slider_controllers.change(function (e) {
        const direct = $("#" + e.target.id.replace("slider", "direct"));
        direct.each((i, elem) => elem.value = e.target.value);

    });

    const direct_controllers = $("[id^=direct-]");
    direct_controllers.attr("step", 0.01);
    direct_controllers.each((i, elem) => {
        const slider = $("#" + elem.id.replace("direct", "slider"));
        elem.value = slider.val();
        const step = $("#" + elem.id.replace("direct", "step"));
        elem.step = step.val();
    } );
    direct_controllers.change(function (e) {
        const slider = $("#" + e.target.id.replace("direct", "slider"));
        slider.each((i, elem) => elem.value = e.target.value);
    });
}

function setup_controllers() {
    $("#file-button-upload").click(
        (e) => $("#modal-file-upload").show()
    );
    $(".custom-file-input").on("change", function(e) {
        const fileNames = Array.from(e.target.files).map(f => f.name).join(", ");
      $(this).siblings(".custom-file-label").addClass("selected").html(fileNames);
    });
    $("#button-upload-save-files").click(
        (e) => {
            e.target.innerHTML = '<span class="spinner-border spinner-border-sm"  id="button-upload-save-files-spinner" role="status" aria-hidden="true"></span> Loading...';

            request_upload();

            e.target.innerHTML = "Save Changes"
            // $("#button-upload-save-files-spinner").show();
        }
    );
    $("#button-upload-close").click(
        (e) => $("#modal-file-upload").hide()
    );
    $(".close").click(
        (e) => $("#modal-file-upload").hide()
    );
}
setup_sliders();
setup_controllers();

function request_motion() {

}

function request_stitch() {
    const request = new Request(API_GATEWAY);

}

function request_focus() {

}

function request_upload() {
    const request = new Request(API_GATEWAY, {method: "POST", mode: 'no-cors', body: {}});
    fetch(request)
      .then(response => {
        if (response.status === 200) {
          return response.json();
        } else {
            console.error(response);
          throw new Error('Something went wrong on api server!');
        }
      })
      .then(response => {
        console.debug(response);
        // ...
      }).catch(error => {
        console.error(error);
  });

}


function upload_handler(e){
    console.log(e);
}


request_upload();





// use document.getElementById('id').innerHTML = 'text' to change text in a paragraph, for example.

// const slider = {
//
//   get_position: function() {
//     var marker_pos = $('#marker1').position();
//     console.log($('#marker1').position());
//     var left_pos = marker_pos.left + slider.marker_size / 2;
//     var top_pos = marker_pos.top + slider.marker_size / 2;
//
//     slider.position = {
//       left: left_pos,
//       top: top_pos,
//       x: Math.round(slider.round_factor.x * (left_pos * slider.xmax / slider.width)) / slider.round_factor.x,
//       y: Math.round((slider.round_factor.y * (slider.height - top_pos) * slider.ymax / slider.height)) / slider.round_factor.y,
//     };
//     return slider.position;
//   },
//
//   display_position: function() {
//     document.getElementById("coord").innerHTML = 'x: ' + slider.position.x.toString() + '<br> y: ' + slider.position.y.toString();
//   },
//
//   draw: function(x_size, y_size, xmax, ymax, marker_size, round_to) {
//       console.log("in draw");
//     if ((x_size === undefined) && (y_size === undefined) && (xmax === undefined) && (ymax === undefined) && (marker_size === undefined) && (round_to === undefined)) {
//       x_size = 150;
//       y_size = 150;
//       xmax = 1;
//       ymax = 1;
//       marker_size = 20;
//       round_to = 2;
//     }
//     slider.marker_size = marker_size;
//     slider.height = y_size;
//     slider.width = x_size;
//     slider.xmax = xmax;
//     slider.ymax = ymax;
//     round_to = Math.pow(10, round_to);
//     slider.round_factor = {
//       x: round_to,
//       y: round_to,
//     };
//
//
//     $("#markerbounds").css({
//       "width": (x_size + marker_size).toString() + 'px',
//       "height": (y_size + marker_size).toString() + 'px',
//     });
//
//     $("#box").css({
//       "width": x_size.toString() + 'px',
//       "height": y_size.toString() + 'px',
//       "top": marker_size / 2,
//       "left": marker_size / 2,
//     });
//
//     $("#marker1").css({
//       "width": marker_size.toString() + 'px',
//       "height": marker_size.toString() + 'px',
//     });
//
//     $("#marker2").css({
//       "width": marker_size.toString() + 'px',
//       "height": marker_size.toString() + 'px',
//     });
//
//     $("#coord").css({
//       "top": x_size + marker_size / 2
//     });
//
//     $("#widget").css({
//       "width": (x_size + marker_size).toString() + 'px',
//     });
//
//     const pos = slider.get_position();
//     slider.display_position();
//     line = $("#slice-line");
//     if (line){
//         line.y1 = pos.y;
//         line.x2 = 100;
//         line.y2 = 100;
//     }
//
//
//   },
//
// };
//
// $("#marker1").draggable({
//   containment: "#markerbounds",
//   drag: function() {
//     const pos = slider.get_position();
//     slider.display_position();
//     const l = $("#slice-line");
//     console.log(l);
//     l.x1 = pos.x;
//     l.y1 = pos.y;
//   },
// });
//
// $("#marker2").draggable({
//   containment: "#markerbounds",
//   drag: function() {
//     slider.get_position();
//     slider.display_position();
//   },
// });
//
// //syntax for rendering is:
// //  slider.render(width, height, width-range, height-range, marker size, output decimal places)
//
// slider.draw(300,100,1,1,10,3);


// check to make sure the defaults work:
//slider.draw();

