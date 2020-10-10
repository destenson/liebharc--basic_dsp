(function() {var implementors = {};
implementors["basic_dsp_vector"] = [{"text":"impl Clone for Real","synthetic":false,"types":[]},{"text":"impl Clone for Complex","synthetic":false,"types":[]},{"text":"impl Clone for RealOrComplex","synthetic":false,"types":[]},{"text":"impl Clone for Time","synthetic":false,"types":[]},{"text":"impl Clone for Freq","synthetic":false,"types":[]},{"text":"impl Clone for TimeOrFreq","synthetic":false,"types":[]},{"text":"impl Clone for MultiCoreSettings","synthetic":false,"types":[]},{"text":"impl&lt;S, T, N, D&gt; Clone for DspVec&lt;S, T, N, D&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: ToSlice&lt;T&gt; + Clone,<br>&nbsp;&nbsp;&nbsp;&nbsp;T: RealNumber,<br>&nbsp;&nbsp;&nbsp;&nbsp;N: NumberSpace + Clone,<br>&nbsp;&nbsp;&nbsp;&nbsp;D: Domain + Clone,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl Clone for ErrorReason","synthetic":false,"types":[]},{"text":"impl Clone for PaddingOption","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;Clone&gt; Clone for Statistics&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl Clone for DataDomain","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;Clone, N:&nbsp;Clone, D:&nbsp;Clone&gt; Clone for TypeMetaData&lt;T, N, D&gt;","synthetic":false,"types":[]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()