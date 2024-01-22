#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "portaudio.h"

#define SAMPLE_RATE  (44100)
#define FRAMES_PER_BUFFER (2048)
#define NUM_SECONDS     (3)
#define NUM_CHANNELS    (2)
#define DITHER_FLAG     (0) 


/* Select sample format. */
#define PA_SAMPLE_TYPE paFloat32;
typedef float SAMPLE;
#define SAMPLE_SILENCE  (0.0f)
#define PRINTF_S_FORMAT "%.8f"

namespace zazamrealtime{

    typedef struct
    {
        int          frameIndex;  /* Index into sample array. */
        int          maxFrameIndex;
        SAMPLE      *recordedSamples;
    } paTestData;

    /**
     * Recorder class contains methods to record from microphone using 
     * PortaAudio library. The code has been adapted form the library examples.
     * @todo: Modernize the code. 
    */
    template<typename Scalar>
    class Recorder{
        public:
            void record(std::vector<Scalar> &audio_output);
        private:
            static int record_callback(const void *inputBuffer, void *outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void *userData);

    };

    template<typename Scalar>
    void Recorder<Scalar>::record(std::vector<Scalar> &audio_output){
        PaStreamParameters  inputParameters,
                        outputParameters;
        PaStream*           stream;
        PaError             err = paNoError;
        paTestData          data;
        int                 i;
        int                 totalFrames;
        int                 numSamples;
        int                 numBytes;
        SAMPLE              max, val;
        double              average;

        printf("patest_record.c\n"); fflush(stdout);

        data.maxFrameIndex = totalFrames = NUM_SECONDS * SAMPLE_RATE; /* Record for a few seconds. */
        data.frameIndex = 0;
        numSamples = totalFrames * NUM_CHANNELS;
        numBytes = numSamples * sizeof(SAMPLE);
        data.recordedSamples = (SAMPLE *) malloc( numBytes ); /* From now on, recordedSamples is initialised. */
        if( data.recordedSamples == NULL )
        {
            printf("Could not allocate record array.\n");
            goto done;
        }
        for( i=0; i<numSamples; i++ ) data.recordedSamples[i] = 0;

        err = Pa_Initialize();
        if( err != paNoError ) goto done;

        inputParameters.device = Pa_GetDefaultInputDevice(); /* default input device */
        if (inputParameters.device == paNoDevice) {
            fprintf(stderr,"Error: No default input device.\n");
            goto done;
        }
        inputParameters.channelCount = 2;                    /* stereo input */
        inputParameters.sampleFormat = PA_SAMPLE_TYPE;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo( inputParameters.device )->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = NULL;

        /* Record some audio. -------------------------------------------- */
        err = Pa_OpenStream(
                &stream,
                &inputParameters,
                NULL,                  /* &outputParameters, */
                SAMPLE_RATE,
                FRAMES_PER_BUFFER,
                paClipOff,      /* we won't output out of range samples so don't bother clipping them */
                record_callback,
                &data );
        if( err != paNoError ) goto done;

        err = Pa_StartStream( stream );
        if( err != paNoError ) goto done;
        printf("\n=== Now recording!! Please speak into the microphone. ===\n"); fflush(stdout);

        while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
        {
            Pa_Sleep(1000);
            printf("index = %d\n", data.frameIndex ); fflush(stdout);
        }
        if( err < 0 ) goto done;

        err = Pa_CloseStream( stream );
        if( err != paNoError ) goto done;

        audio_output.insert(audio_output.end(), data.recordedSamples, data.recordedSamples+numSamples);
        
        return;
    done:
        Pa_Terminate();
        if( data.recordedSamples )       /* Sure it is NULL or valid. */
            free( data.recordedSamples );
        if( err != paNoError )
        {
            fprintf( stderr, "An error occurred while using the portaudio stream\n" );
            fprintf( stderr, "Error number: %d\n", err );
            fprintf( stderr, "Error message: %s\n", Pa_GetErrorText( err ) );
            err = 1;          /* Always return 0 or 1, but no other return codes. */
        }
        abort();

    }   


    template<typename Scalar>
    int Recorder<Scalar>::record_callback( const void *inputBuffer, void *outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void *userData )
    {
        paTestData *data = (paTestData*)userData;
        const SAMPLE *rptr = (const SAMPLE*)inputBuffer;
        SAMPLE *wptr = &data->recordedSamples[data->frameIndex * NUM_CHANNELS];
        long framesToCalc;
        long i;
        int finished;
        unsigned long framesLeft = data->maxFrameIndex - data->frameIndex;


        if( framesLeft < framesPerBuffer )
        {
            framesToCalc = framesLeft;
            finished = paComplete;
        }
        else
        {
            framesToCalc = framesPerBuffer;
            finished = paContinue;
        }

        if( inputBuffer == NULL )
        {
            for( i=0; i<framesToCalc; i++ )
            {
                *wptr++ = SAMPLE_SILENCE;  /* left */
                if( NUM_CHANNELS == 2 ) *wptr++ = SAMPLE_SILENCE;  /* right */
            }
        }
        else
        {
            for( i=0; i<framesToCalc; i++ )
            {
                *wptr++ = *rptr++;  /* left */
                if( NUM_CHANNELS == 2 ) *wptr++ = *rptr++;  /* right */
            }
        }
        data->frameIndex += framesToCalc;
        return finished;
    }
}