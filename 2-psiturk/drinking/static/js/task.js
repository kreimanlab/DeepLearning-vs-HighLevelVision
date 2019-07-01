/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

var mycondition = condition;  // these two variables are passed by the psiturk server process
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to
// they are not used in the stroop code but may be useful to you

// All pages to be loaded
var pages = [
        "instructions/instruct-1.html",
        "instructions/instruct-ready.html",
        "stage.html",
        "postquestionnaire.html"
];

psiTurk.preloadPages(pages);

var instructionPages = [ // add as a list as many pages as you like
        "instructions/instruct-1.html",
        "instructions/instruct-ready.html"
];


/********************
* HTML manipulation
*
* All HTML files in the templates directory are requested
* from the server when the PsiTurk object is created above. We
* need code to get those pages from the PsiTurk object and
* insert them into the document.
*
********************/

/********************
* STROOP TEST       *
********************/

var StroopExperiment = function() {

        psiTurk.recordUnstructuredData("mode", mode);

        var wordon; // time word is presented
        const TotalNumImage = 778;
        var task_idx = -1; // because of asynchronous js. d3.csv is executed after everything else in function() next.
        var img_name = "";

        //psiTurk.preloadImages(imagelist);

        // Stimuli for a basic Stroop experiment        
        psiTurk.recordUnstructuredData("condition", condition);

        var next = function() {
                task_idx = task_idx+1;
                if (task_idx < TotalNumImage) {
                        //window.alert(task_idx);
                        d3.csv("csv/s1gifDrinkFull-shuf.csv", function(data) {
                                d3.select("#stim").html('<img src='+data[task_idx].image_url+' alt="stimuli" style="width:100%">');
                                img_name = data[task_idx].image_url;
                        });
                        wordon = new Date().getTime();
                }
                else {
                        finish();
                }
        };

        var finish = function() {
            currentview = new Questionnaire();
        };

        // Load the stage.html snippet into the body of the page
        psiTurk.showPage('stage.html');

        // Start the test; initialize everything
        next();
        document.getElementById("submittrial").addEventListener("click", mengmiClick);


        function mengmiClick()
        {
            var response = document.querySelector('input[name="categories"]:checked').value;
                //var response = document.getElementById("response").value;
            //document.getElementById("response").value = "";

            if (response.length > 0)
            {
                var rt = new Date().getTime() - wordon;
                //document.getElementById("demo").innerHTML = response;

                psiTurk.recordTrialData({'phase':"TEST",
                    'image_name':img_name,
                    'response':response, //worker response for image name 
                    'rt':rt, //response time
                    'trial': task_idx+1} //trial index starting from 1
                );
                next();
                document.querySelector('input[name="categories"]:checked').checked = false;
            }
            else
            {
                window.alert("No answer selected.");
            }
        }
};


/****************
* Questionnaire *
****************/

var Questionnaire = function() {

        var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

        record_responses = function() {

                psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'submit'});

                $('input').each( function(i, val) {
                        psiTurk.recordUnstructuredData(this.id, this.value);
                });

        };

        prompt_resubmit = function() {
                document.body.innerHTML = error_message;
                $("#resubmit").click(resubmit);
        };

        resubmit = function() {
                document.body.innerHTML = "<h1>Trying to resubmit...</h1>";
                reprompt = setTimeout(prompt_resubmit, 10000);

                psiTurk.saveData({
                        success: function() {
                            clearInterval(reprompt);
                                psiTurk.completeHIT();
                        },
                        error: prompt_resubmit
                });
        };


        // Load the questionnaire snippet
        psiTurk.showPage('postquestionnaire.html');
        psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'begin'});

        $("#next").click(function () {
            record_responses();
            psiTurk.saveData({
            success: function(){
                psiTurk.completeHIT(); // when finished saving compute bonus, the quit
            },
            error: prompt_resubmit});
        });


};

// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
$(window).load( function(){
    psiTurk.doInstructions(
        instructionPages, // a list of pages you want to display in sequence
        function() { currentview = new StroopExperiment(); } // what you want to do when you are done with instructions
    );
});
