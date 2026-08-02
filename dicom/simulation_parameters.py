class Patient_parameters:
    '''
    Patient class includes patient biological data
    '''
    def __init__(self, gender, tumor_site, tumor_volume_fraction, relative_blood_density, relative_perfusion, organs):
        self.gender = gender
        self.tumor_site = tumor_site
        self.tumor_volume_fraction = tumor_volume_fraction
        self.relative_blood_density = relative_blood_density
        self.relative_perfusion = relative_perfusion
        self.organs = organs

        if gender.lower() in ['m', 'male']:
            self.sheet_name = 'male_new'
            self.TBV = 5.3  # 5.3 L total simulation volume.
            self.CO = 6.5 / 60  # 6.5 L/min cardiac output.

        elif gender.lower() in ['f', 'female']:
            self.sheet_name = 'female_new'
            self.TBV = 3.9  # 3.9 L total simulation volume.
            self.CO = 5.9 / 60  # 5.9 L/min cardiac output.
        else:
            raise ValueError('Cannot deduce gender.')

    def __getitem__(self, key):
        return self.to_dict()[key]

    def summary(self):
        print('Gender: {}'.format(self.gender))
        print('Sheet Name: {}'.format(self.sheet_name))
        print('Tumor site: {}'.format(self.tumor_site))
        print("Organs: {}".format(self.organs))
        print("Total Blood Count: {}".format(self.relative_blood_density))
        print("Cardiac Output: {}".format(self.CO))

    def to_dict(self):
        return {
            "gender": self.gender,
            "sheet_name": self.sheet_name,
            "tumor_site": self.tumor_site,
            "tumor_volume_fraction": self.tumor_volume_fraction,
            "relative_blood_density": self.relative_blood_density,
            "relative_perfusion": self.relative_perfusion,
            "organs": self.organs,
            "TBV": self.TBV,
            "CO": self.CO
        }
class Simulation_parameters:
    '''
    Simulation class includes siumation parameters
    '''
    def __init__(self,sample_size,nr_steps,dt,weibull_shape,generate_new,random_walk,accumulate):
        self.sample_size = sample_size #number of simulation particles
        self.nr_steps = nr_steps #number of time steps
        self.dt = dt #in seconds
        self.weibull_shape = weibull_shape #weibull distribution parameter
        self.generate_new = generate_new
        self.random_walk = random_walk
        self.accumulate = accumulate

    def __getitem__(self, key):
        return self.to_dict()[key]

    def summary(self):
        print('Sample Size: {}'.format(self.sample_size))
        print('Nr steps: {}'.format(self.nr_steps))
        print('dt: {}'.format(self.dt))
        print('Weibull shape: {}'.format(self.weibull_shape))
        print('Generate New: {}'.format(self.generate_new))
        print('Random walk: {}'.format(self.random_walk))
        print('Accumulate: {}'.format(self.accumulate))

    def to_dict(self):
        return {
            "sample_size": self.sample_size,
            "nr_steps": self.nr_steps,
            "dt": self.dt,
            "weibull_shape": self.weibull_shape,
            "generate_new": self.generate_new,
            "random_walk": self.random_walk,
            "accumulate": self.accumulate
        }
class Treatment_parameters:
    '''
    Parameters class includes parameters for the treatment
    '''
    def __init__(self,nr_fractions,total_beam_on_time,start_times,beam_on_times):
        self.nr_fractions = nr_fractions
        self.total_beam_on_time = total_beam_on_time
        self.start_times = start_times
        self.beam_on_times = beam_on_times
        assert(sum(beam_on_times) == total_beam_on_time), 'Beam-on-time of separate fields should equal total beam-on-time.'
        assert(start_times[:-1] + beam_on_times[:-1] <= start_times[1:]), 'Cannot start new field before completing current.'

    def __getitem__(self, key):
        return self.to_dict()[key]

    def to_dict(self):
        return {
        "nr_fractions": self.nr_fractions,
        "total_beam_on_time": self.total_beam_on_time,
        "start_times": self.start_times,
        "beam_on_times": self.beam_on_times
        }

    def summary(self):
        print('nr_fractions: {}'.format(self.nr_fractions))
        print('total_beam_on_time: {}'.format(self.total_beam_on_time))
        print('start_times: {}'.format(self.start_times))
        print('beam_on_times: {}'.format(self.beam_on_times))
