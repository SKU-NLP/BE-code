package com.app.NLP.global.exception;

import com.app.NLP.global.template.RspTemplate;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingRequestHeaderException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.method.annotation.HandlerMethodValidationException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Slf4j
@RestControllerAdvice
public class ControllerExceptionAdvice {

    /**
     * Custom Exception
     */
    @ExceptionHandler(CustomException.class)
    public ResponseEntity<RspTemplate<Void>> handleCustomException(CustomException e) {
        return ResponseEntity
                .status(e.getHttpStatus())
                .body(RspTemplate.error(e.getError(), e.getMessage()));
    }

    /**
     * @Valid (RequestBody) Validation Error
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<RspTemplate<Void>> handleMethodArgumentNotValid(MethodArgumentNotValidException e) {
        FieldError fieldError = e.getBindingResult().getFieldError();
        String message = (fieldError != null)
                ? fieldError.getDefaultMessage()
                : Error.BAD_REQUEST_VALIDATION.getMessage();

        return ResponseEntity
                .status(Error.BAD_REQUEST_VALIDATION.getErrorCode())
                .body(RspTemplate.error(Error.BAD_REQUEST_VALIDATION, message));
    }

    /**
     * @Validated (RequestParam/PathVariable) Validation Error (Spring Boot 3)
     */
    @ExceptionHandler(HandlerMethodValidationException.class)
    public ResponseEntity<RspTemplate<Void>> handleHandlerMethodValidation(HandlerMethodValidationException e) {
        return ResponseEntity
                .status(Error.BAD_REQUEST_VALIDATION.getErrorCode())
                .body(RspTemplate.error(Error.BAD_REQUEST_VALIDATION, Error.BAD_REQUEST_VALIDATION.getMessage()));
    }

    /**
     * Wrong HTTP Method
     */
    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    public ResponseEntity<RspTemplate<Void>> handleMethodNotSupported(HttpRequestMethodNotSupportedException e) {
        return ResponseEntity
                .status(Error.INVALID_REQUEST.getErrorCode())
                .body(RspTemplate.error(Error.INVALID_REQUEST, Error.INVALID_REQUEST.getMessage()));
    }

    /**
     * Unsupported Media Type
     */
    @ExceptionHandler(HttpMediaTypeNotSupportedException.class)
    public ResponseEntity<RspTemplate<Void>> handleMediaTypeNotSupported(HttpMediaTypeNotSupportedException e) {
        return ResponseEntity
                .status(Error.INVALID_REQUEST.getErrorCode())
                .body(RspTemplate.error(Error.INVALID_REQUEST, Error.INVALID_REQUEST.getMessage()));
    }

    /**
     * Missing Request Param
     */
    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResponseEntity<RspTemplate<Void>> handleMissingParam(MissingServletRequestParameterException e) {
        return ResponseEntity
                .status(Error.BAD_REQUEST_VALIDATION.getErrorCode())
                .body(RspTemplate.error(Error.BAD_REQUEST_VALIDATION, Error.BAD_REQUEST_VALIDATION.getMessage()));
    }

    /**
     * Missing Request Header
     */
    @ExceptionHandler(MissingRequestHeaderException.class)
    public ResponseEntity<RspTemplate<Void>> handleMissingHeader(MissingRequestHeaderException e) {
        return ResponseEntity
                .status(Error.BAD_REQUEST_VALIDATION.getErrorCode())
                .body(RspTemplate.error(Error.BAD_REQUEST_VALIDATION, Error.BAD_REQUEST_VALIDATION.getMessage()));
    }

    /**
     * JSON Parse / Body Missing / Type mismatch (RequestBody)
     */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<RspTemplate<Void>> handleNotReadable(HttpMessageNotReadableException e) {
        return ResponseEntity
                .status(Error.BAD_REQUEST_VALIDATION.getErrorCode())
                .body(RspTemplate.error(Error.BAD_REQUEST_VALIDATION, Error.BAD_REQUEST_VALIDATION.getMessage()));
    }

    /**
     * Fallback - 500
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<RspTemplate<Void>> handleException(Exception e, HttpServletRequest request) {
        log.error("UNHANDLED EXCEPTION: {} {}", request.getMethod(), request.getRequestURI(), e);
        return ResponseEntity
                .status(Error.INTERNAL_SERVER_ERROR.getErrorCode())
                .body(RspTemplate.error(Error.INTERNAL_SERVER_ERROR, Error.INTERNAL_SERVER_ERROR.getMessage()));
    }
}
