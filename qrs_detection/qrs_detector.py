from ecgdetectors import Detectors


def return_r_peaks(signal):

    detector = Detectors(50)

    r_peaks = detector.pan_tompkins_detector(signal)
    
    return r_peaks