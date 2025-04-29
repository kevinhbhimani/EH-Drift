import subprocess

# List of detectors to run the jobs for
detectors = ['P00698A'] #V07647A P00698A
# Path to the job submission script
job_script = "/nas/longleaf/home/kbhimani/siggen_ccd/jobs/job.sh"

def submit_job(detector):
    """
    Submits a job to the SLURM scheduler using sbatch.
    
    Parameters:
        detector (str): The name of the detector for which to run the simulation.
    """
    # Command to submit the job using sbatch
    cmd = ['sbatch', job_script, detector]
    
    # Print the command for verification
    print(f"Submitting job for detector: {detector}")
    print("Command:", ' '.join(cmd))
    
    # Run the sbatch command
    subprocess.run(cmd)

def main():
    # Loop through each detector and submit a job
    for detector in detectors:
        submit_job(detector)

if __name__ == "__main__":
    main()
