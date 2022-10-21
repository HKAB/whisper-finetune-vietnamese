//webkitURL is deprecated but nevertheless 
URL = window.URL || window.webkitURL;
var gumStream;
//stream from getUserMedia() 
var rec;
//Recorder.js object 
var input;
//MediaStreamAudioSourceNode we'll be recording 
// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext = new AudioContext;
//new audio context to help us record 
var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");
//add events to those 3 buttons 
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);

function startRecording() { 
    var constraints = {
        audio: true,
        video: false
    } 
    /* Disable the record button until we get a success or fail from getUserMedia() */
    
    recordButton.disabled = true;
    stopButton.disabled = false;
    pauseButton.disabled = false
    
    /* We're using the standard promise based getUserMedia()
    
    https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia */
    
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ..."); 
        /* assign to gumStream for later use */
        gumStream = stream;
        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);
        /* Create the Recorder object and configure to record mono sound (1 channel) Recording 2 channels will double the file size */
        rec = new Recorder(input, {
            numChannels: 1
        }) 
        //start the recording process 
        rec.record()
        console.log("Recording started");
    }).catch(function(err) {
        //enable the record button if getUserMedia() fails 
        recordButton.disabled = false;
        stopButton.disabled = true;
        pauseButton.disabled = true
    });
}

function pauseRecording() {
    console.log("pauseButton clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause 
        rec.stop();
        pauseButton.innerHTML = "Resume";
    } else {
        //resume 
        rec.record()
        pauseButton.innerHTML = "Pause";
    }
}

function stopRecording() {
    console.log("stopButton clicked");
    //disable the stop button, enable the record too allow for new recordings 
    stopButton.disabled = true;
    recordButton.disabled = false;
    pauseButton.disabled = true;
    //reset button just in case the recording is stopped while paused 
    pauseButton.innerHTML = "Pause";
    //tell the recorder to stop the recording 
    rec.stop(); //stop microphone access 
    gumStream.getAudioTracks()[0].stop();
    //create the wav blob and pass it on to createDownloadLink 
    rec.exportWAV(createDownloadLink);
}

var response_text;

function createDownloadLink(blob) {
    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');
    //add controls to the <audio> element 
    au.controls = true;
    au.src = url;
    //link the a element to the blob 
    link.href = url;
    link.download = new Date().toISOString() + '.wav';
    link.innerHTML = link.download;
    //add the new audio and a elements to the li element 
    li.appendChild(au);
    // li.appendChild(link);
    //add the li element to the ordered list 




    var filename = new Date().toISOString();
    //filename to send to server without extension 
    //upload link 
    var eng = document.createElement('button');
    eng.setAttribute("id",'translate');
    var viet = document.createElement('button');
    viet.setAttribute("id",'translate');
    var text = document.createElement('span');
    text.innerHTML = "";
    eng.href = "#";
    eng.innerHTML = "Transcribe to English";
    eng.addEventListener("click", function(event) {
        var xhr = new XMLHttpRequest();
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "english", true);
        xhr.send(fd);
        
        xhr.onload = () => {
            const response = xhr.responseText;
            const reader = new FileReader();
            reader.onload = function(){
                text.innerHTML = response;
            }
            reader.readAsArrayBuffer(blob);
        };
        
        
    });
    
    viet.href = "#";
    viet.innerHTML = "Transcribe to Vienamese";
    viet.addEventListener("click", function(event) {
        var xhr = new XMLHttpRequest();
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "vietnamese", true);
        xhr.send(fd);
        
        xhr.onload = () => {
            const response = xhr.responseText;
            const reader = new FileReader();
            reader.onload = function(){
                text.innerHTML = response;
            }
            reader.readAsArrayBuffer(blob);
        };
        
        
    });

    // const text2 = document.createElement('span');
    // text2.innerHTML = response_text;

    li.appendChild(document.createTextNode(" ")) //add a space in between 
    li.appendChild(eng) //add the upload link to li
    li.appendChild(document.createTextNode(" ")) //add a space in between 
    li.appendChild(viet) //add the upload link to li
    li.appendChild(document.createTextNode(" "))
    li.appendChild(text);
    
    recordingsList.appendChild(li);
    
}
