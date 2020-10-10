initSidebarItems({"constant":[["STATS_VEC_CAPACTIY","The maximum `len` for any of the `*split` methods."]],"enum":[["DataDomain","The domain of a data vector"],["ErrorReason","Enumeration of all error reasons"],["PaddingOption","An option which defines how a vector should be padded"]],"fn":[["print_calibration","Prints debug information about the calibration. The calibration determines when the library will start to spawn threads. If a calibration hasn't been performed yet than calling this function will trigger the calibration."]],"mod":[["conv_types","Types around a convolution, see also https://en.wikipedia.org/wiki/Convolution."],["meta",""],["numbers","Traits from the `num` crate which are used inside `basic_dsp` and extensions to those traits."],["window_functions","This mod contains a definition for window functions and provides implementations for a few standard windows. See the `WindowFunction` type for more information."]],"struct":[["DspVec","A 1xN (one times N elements) or Nx1 data vector as used for most digital signal processing (DSP) operations."],["FixedLenBuffer","A buffer which gets initalized with a data storage type and then always keeps that."],["FixedLenBufferBurrow","Buffer borrow type for `SingleBuffer`."],["MultiCoreSettings","Holds parameters which specify how multiple cores are used to execute an operation."],["NoBuffer","This type can be used everytime the API asks for a buffer to disable any buffering."],["NoBufferBurrow","Buffer borrow type for `NoBuffer`."],["NoTradeBufferBurrow","Buffer borrow type for `NoTradeBufferBurrow`."],["SingleBuffer","A buffer which stores a single vector and never shrinks."],["SingleBufferBurrow","Buffer borrow type for `SingleBuffer`."],["Statistics","Statistics about numeric data"],["TypeMetaData","Holds meta data about a type."]],"trait":[["ApproximatedOps","Recommended to be only used with the CPU feature flags `sse` or `avx`."],["Buffer","A buffer which can be used by other types. Types will call buffers to create new arrays. A buffer may can implement any buffering strategy."],["BufferBorrow","A \"slice-like\" type which also allows to"],["ComplexIndex","Like `std::ops::Index` but with a different method name so that it can be used to implement an additional range accessor for complex data."],["ComplexIndexMut","Like `std::ops::IndexMut` but with a different method name so that it can be used to implement a additional range accessor for complex data."],["ComplexNumberSpace","Trait for types containing complex data."],["ComplexOps","Operations on complex types."],["ComplexToRealGetterOps","Defines getters to get real data from complex types."],["ComplexToRealSetterOps","Defines setters to create complex data from real data."],["ComplexToRealTransformsOps","Defines transformations from complex to real number space."],["ComplexToRealTransformsOpsBuffered","Defines transformations from complex to real number space."],["Convolution","Provides a convolution operations."],["ConvolutionOps","Provides a convolution operation for types which at some point are slice based."],["CrossCorrelationArgumentOps","This trait allows to transform an argument so that it can be used for cross correlation. Refer to the description of `CrossCorrelationOps` for more details."],["CrossCorrelationOps","Cross-correlation of data vectors. See also https://en.wikipedia.org/wiki/Cross-correlation"],["DiffSumOps","A trait to calculate the diff (1st derivative in a discrete number space) or cumulative sum (integral  in a discrete number space)."],["Domain","Domain (time or frequency) information."],["DotProductOps","An operation which multiplies each vector element with a constant"],["ElementaryOps","Elementary algebra on types: addition, subtraction, multiplication and division"],["ElementaryWrapAroundOps","Elementary algebra on types where the argument might contain less data points than `self`."],["FloatIndex","Like `std::ops::Index` but with a different method name so that it can be used to implement an additional range accessor for float data."],["FloatIndexMut","Like `std::ops::IndexMut` but with a different method name so that it can be used to implement a additional range accessor for float data."],["FrequencyDomain","Trait for types containing frequency domain data."],["FrequencyDomainOperations","Defines all operations which are valid on `DataVecs` containing frequency domain data."],["FrequencyMultiplication","Provides a frequency response multiplication operations."],["FrequencyToTimeDomainOperations","Defines all operations which are valid on `DataVecs` containing frequency domain data."],["FromVector","Retrieves the underlying storage from a vector."],["FromVectorFloat","Retrieves the underlying storage from a vector. Returned value will always hold floating point numbers."],["GetMetaData","Gets the meta data of a type. This can be used to create a new type with the same meta data."],["InsertZerosOps","A trait to insert zeros into the data at some specified positions."],["InsertZerosOpsBuffered","A trait to insert zeros into the data at some specified positions. A buffer is used for types which can't be resized and/or to speed up the calculation."],["InterleaveToVector","Conversion from two instances of a generic data type into a dsp vector with complex data."],["InterpolationOps","Provides interpolation operations for real and complex data vectors."],["MapAggregateOps","Operations which allow to iterate over the vector and to derive results."],["MapInplaceOps","Operations which allow to iterate over the vector and to derive results or to change the vector."],["MergeOps","Merges several pieces of equal size into one data chunk."],["MetaData","A trait which provides information about number space and domain."],["ModuloOps","Operations on real types."],["NumberSpace","Number space (real or complex) information."],["OffsetOps","An operation which adds a constant to each vector element"],["PosEq","Expresses at compile time that two classes could potentially represent the same number space or domain."],["PowerOps","Roots, powers, exponentials and logarithms."],["PreciseDotProductOps","An operation which multiplies each vector element with a constant"],["PreciseStatisticsOps","Offers the same functionality as the `StatisticsOps` trait but the statistics are calculated in a more precise (and slower) way."],["PreciseStatisticsSplitOps","Offers the same functionality as the `StatisticsOps` trait but the statistics are calculated in a more precise (and slower) way."],["PreciseStats","A trait for statistics which allows to add new values in a way so that the numerical uncertainty has less impact on the final results."],["PreciseSumOps","Offers the same functionality as the `SumOps` trait but the sums are calculated in a more precise (and slower) way."],["RealInterpolationOps","Provides interpolation operations which are only applicable for real data vectors."],["RealNumberSpace","Trait for types containing real data."],["RealOps","Operations on real types."],["RealToComplexTransformsOps","Defines transformations from real to complex number space."],["RealToComplexTransformsOpsBuffered","Defines transformations from real to complex number space."],["RededicateForceOps","This trait allows to change a data type and performs the Conversion without any checks. `RededicateOps` provides the same functionality but performs runtime checks to avoid that data is interpreted the wrong way."],["RededicateOps","This trait allows to change a data type. The operations will convert a type to a different one and set `self.len()` to zero. However `self.allocated_len()` will remain unchanged. The use case for this is to allow to reuse the memory of a vector for different operations."],["RededicateToOps","This trait allows to change a data type. The operations will convert a type to a different one and set `self.len()` to zero. However `self.allocated_len()` will remain unchanged. The use case for this is to allow to reuse the memory of a vector for different operations."],["ReorganizeDataOps","This trait allows to reorganize the data by changing positions of the individual elements."],["Resize","A trait for storage types which are known to have the capability to increase their capacity."],["ResizeBufferedOps","Operations to resize a data type."],["ResizeOps","Operations to resize a data type."],["ScaleOps","An operation which multiplies each vector element with a constant"],["SplitOps","Splits the data into several smaller pieces of equal size."],["StatisticsOps","This trait offers operations to calculate statistics about the data in a type."],["StatisticsSplitOps","This trait offers operations to calculate statistics about the data in a type."],["Stats","Operations on statistics."],["SumOps","Offers operations to calculate the sum or the sum of squares."],["SymmetricFrequencyToTimeDomainOperations","Defines all operations which are valid on `DataVecs` containing frequency domain data and the data is assumed to half of complex conjugate symmetric spectrum round 0 Hz where the 0 Hz element itself is real."],["SymmetricTimeToFrequencyDomainOperations","Defines all operations which are valid on `DataVecs` containing real time domain data."],["TimeDomain","Trait for types containing time domain data."],["TimeDomainOperations","Defines all operations which are valid on `DataVecs` containing time domain data."],["TimeToFrequencyDomainOperations","Defines all operations which are valid on `DataVecs` containing time domain data."],["ToComplexResult","Specifies what the the result is if a type is transformed to complex numbers."],["ToComplexVector","Conversion from a generic data type into a dsp vector with complex data."],["ToComplexVectorPar","Conversion from a generic data type into a dsp vector with complex data."],["ToDspVector","Conversion from a generic data type into a dsp vector which tracks its meta information (domain and number space) only at runtime. See `ToRealVector` and `ToComplexVector` for alternatives which track most of the meta data with the type system and therefore avoid runtime errors."],["ToDspVectorPar","Conversion from a generic data type into a dsp vector which tracks its meta information (domain and number space) only at runtime. See `ToRealVector` and `ToComplexVector` for alternatives which track most of the meta data with the type system and therefore avoid runtime errors."],["ToFreqResult","Specifies what the the result is if a type is transformed to frequency domain."],["ToRealResult","Specifies what the the result is if a type is transformed to real numbers."],["ToRealTimeResult","Specifies what the the result is if a type is transformed to real numbers in time domain."],["ToRealVector","Conversion from a generic data type into a dsp vector with real data."],["ToRealVectorPar","Conversion from a generic data type into a dsp vector with real data."],["ToSlice","A trait to convert a type into a slice."],["ToSliceMut","A trait to convert a type into a mutable slice."],["ToTimeResult","Specifies what the the result is if a type is transformed to time domain."],["TrigOps","Trigonometry methods."],["Vector","A trait for vector types."]],"type":[["ComplexFreqVec","A vector with complex numbers in frequency domain."],["ComplexFreqVec32","A vector with complex numbers in frequency domain."],["ComplexFreqVec64","A vector with complex numbers in frequency domain."],["ComplexFreqVecSlice32","A vector with complex numbers in frequency domain."],["ComplexFreqVecSlice64","A vector with complex numbers in frequency domain."],["ComplexTimeVec","A vector with complex numbers in time domain."],["ComplexTimeVec32","A vector with complex numbers in time domain."],["ComplexTimeVec64","A vector with complex numbers in time domain."],["ComplexTimeVecSlice32","A vector with complex numbers in time domain."],["ComplexTimeVecSlice64","A vector with complex numbers in time domain."],["GenDspVec","A vector with no information about number space or domain at compile time."],["GenDspVec32","A vector with no information about number space or domain at compile time."],["GenDspVec64","A vector with no information about number space or domain at compile time."],["GenDspVecSlice32","A vector with no information about number space or domain at compile time."],["GenDspVecSlice64","A vector with no information about number space or domain at compile time."],["RealFreqVec","A vector with real numbers in frequency domain."],["RealFreqVec32","A vector with real numbers in frequency domain."],["RealFreqVec64","A vector with real numbers in frequency domain."],["RealFreqVecSlice32","A vector with real numbers in frequency domain."],["RealFreqVecSlice64","A vector with real numbers in frequency domain."],["RealTimeVec","A vector with real numbers in time domain."],["RealTimeVec32","A vector with real numbers in time domain."],["RealTimeVec64","A vector with real numbers in time domain."],["RealTimeVecSlice32","A vector with real numbers in time domain."],["RealTimeVecSlice64","A vector with real numbers in time domain."],["ScalarResult","Scalar result or a reason in case of an error."],["StatsVec","Alias for a vector of any statistical information."],["TransRes","Result for operations which transform a type (most commonly the type is a vector). On success the transformed type is returned. On failure it contains an error reason and the original type with with invalid data which still can be used in order to avoid memory allocation."],["VoidResult","Void/nothing in case of success or a reason in case of an error."]]});